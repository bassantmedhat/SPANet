# SPANet test code

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from collections import OrderedDict

import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import model

import open_clip
import  torch.optim as optim
import torch.nn as nn

from src.loss.loss import ClusterPatch, SeparationPatch, L_norm, CeLoss

from as_dataloader_f import get_as_dataloader

import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import balanced_accuracy_score, f1_score

import torch

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example class labels
class_names = ["Normal", "Mild", "Severe"]

# Explicitly use 'cuda:0' for the visible GPU 6
device = torch.device('cuda:0')  # 'cuda:0' will refer to GPU 6

data_config = {
        "name": "as_tom",
        "data_info_file": 'data/as_tom/annotations-all_NoLeak.csv',
        "dataset_root": 'data/as_tom',
        "text_encoder_model_name": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "text_encoder_cache_dir": "pretrained_models/nlp_model_cache",
        "sample_size": None,
        "sampler": "AS",  # one of 'AS', 'random', 'bicuspid'
        "label_scheme_name": "tufts",
        # one of 'binary', 'all', 'not_severe', 'as_only', 'mild_moderate', 'moderate_severe'
        "view": "all",  # one of  psax, plax, all
        "normalize": True,
        "augmentation": True,
        "img_size": 224,
        "frames": 1,
        "transform_min_crop_ratio": 0.7,
        "transform_rotate_degrees": 15,
        "batch_size": 16,
        "num_workers": 0,
        "iterate_intervals": True,
        # "interval_unit": "cycle",
        "interval_quant": 1.0,
        "interval_unit" : "image",
    }

train_loader = get_as_dataloader(
    config = data_config,
    split = 'train',
    mode = 'train'
)

val_loader =  get_as_dataloader(
    config = data_config,
    split = 'val',
    mode = 'val'
)

test_settings = {

    'ViT-B/16': {
        'base_architecture': 'clip_vitb16',
        'model_path': './my_models/CUB_ViTB16.pth',
        'prototype_shape': (30, 768, 1, 1),
    },
}


wandb.init(
        project="SPANET",
        name="trial",
    )

for model_name, setting in test_settings.items():
    epochs = 30 
    base_architecture = setting['base_architecture']
    model_path = setting['model_path']
    prototype_shape = setting['prototype_shape']

    # inference settings
    batch_size = 16

    # model settings
    img_size = 224
    num_classes = 3
    prototype_activation_function = 'log'
    add_on_layers_type = 'regular'



    sprnet = model.construct_SPRNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)


    sprnet = sprnet.to(device)
    lr = 1e-4
    optimizer = optim.Adam(sprnet.parameters(), lr=lr)

    coefs ={'crs_ent': 1,
            'clst': 0,
            'sep': 0 ,
            'l1':  0
            }

    ce_loss= CeLoss(loss_weight =  coefs['crs_ent'])
    Cluster = ClusterPatch(num_classes=num_classes,loss_weight =  coefs['clst'])
    Separation = SeparationPatch(num_classes=num_classes, loss_weight = coefs['sep'])

    # regularizations for classification layer
    Lnorm_fc = L_norm(mask=1 - torch.t(sprnet.prototype_class_identity), loss_weight = coefs['l1'])



    
    wandb.config.update({
    "optimizer":optimizer
    })


    # Ensure all loss values are converted correctly
    def safe_tolist(value):
        if isinstance(value, (int, float)):  # Scalars don't have `.tolist()`
            return value  # Keep as-is (wandb can log scalars directly)
        elif hasattr(value, "tolist"):  # NumPy arrays or PyTorch tensors
            return value.tolist()
        return value  # Keep unchanged if already a list or other compatible type
    

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))


    for epoch in range(epochs):
        sprnet.train()
        train_loss = 0.0
        n_examples = 0
        n_correct = 0
        n_batches = 0
        total_cross_entropy = 0
        total_cluster_cost = 0
        total_separation_cost = 0
        total_avg_separation_cost = 0
        total_l1 = 0
        best_val_loss = 0 
        all_labels = []
        all_predictions = []

        for batch in tqdm(train_loader):
            
            # input image and target is the class predicition
            input, target = batch['cine'],  batch['target_AS']
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # forward 
            grad_req = torch.enable_grad() 
            with grad_req:
            
                output, min_distances, global_features, local_features = sprnet.image_embed(input)

                cross_entropy = ce_loss.compute(output, target)

                cluster_cost = Cluster.compute(min_distances, target)
                
                separation_cost =  Separation.compute(min_distances, target)

                l1 = sprnet.last_layer.weight.norm(p=1)
   
                _, predicted = torch.max(output.data, 1)

                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()

                n_batches += 1
                total_cross_entropy += cross_entropy.item()
                total_cluster_cost += cluster_cost.item()
                total_separation_cost += separation_cost.item()

                total_l1 += l1.item()
                
                loss = (cross_entropy
                            + cluster_cost
                            + separation_cost
                            + l1)
            
                train_loss += loss.item()
                all_labels.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())


                # backward 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



            del input
            del target
            del output
            del predicted
            del min_distances

            


        train_loss /= n_batches
        total_cross_entropy  = total_cross_entropy/ n_batches
        total_cluster_cost = total_cluster_cost /n_batches
        total_separation_cost = total_separation_cost/ n_batches
        total_l1 = total_l1 / n_batches

        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        avg_f1 = f1_score(all_labels, all_predictions, average='macro')

        print(f"EPOCH {(epoch+1)} /{epochs} train loss: {train_loss}")
        # Save checkpoint for each epoch
        torch.save({'epoch': epoch+1, 'model_state_dict': sprnet.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f"trial/checkpoint_epoch_{epoch+1}.pth")

        wandb.init("trial")

        wandb.log(
            {   "Train cross_entropy" :total_cross_entropy,
                "Train cluster_cost": total_cluster_cost,
                "Train separation_cost": total_separation_cost,
                "Train fc_lnorm": total_l1,
                "Train Balaced Acc":balanced_acc,
                "Train f1_score":avg_f1,
                "Total Train Loss": train_loss,
                "Epoch": epoch,
            }
        )


        # ---- Live Plotting ----
        ax.clear()
        ax.plot(epoch, total_cross_entropy, label="Train cross_entropy", marker="o")
        ax.plot(epoch, total_cluster_cost, label="Train cluster_cost", marker="s")
        ax.plot(epoch, total_separation_cost, label= "Train separation_cost", marker="^")
        ax.plot(epoch, total_l1, label="Train fc_lnorm", marker="d"),
        ax.plot(epoch, balanced_acc, label="Train Balaced Acc", marker="d"),
        ax.plot(epoch, avg_f1, label="Train f1_score", marker="d"),
        ax.plot(epoch, train_loss, label="Total Train Loss", linestyle='-', marker='o')


        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves")
        ax.legend()
        ax.grid(True)

        # Convert matrix to WandB format
        cm = confusion_matrix(all_labels, all_predictions)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")

        # Log
        wandb.log({f"confusion_matrix_EPOCH {(epoch+1)}": wandb.Image(plt)})
        
        plt.pause(0.1)  # Small delay to update the figure

########### validation ############### 
        # Validation Phase
        sprnet.eval()
        val_loss = 0 
        val_total_loss = 0
        val_batch = 0
        val_n_examples = 0
        val_total_cross_entropy = 0
        val_total_cluster_cost = 0
        val_total_separation_cost = 0
        val_total_avg_separation_cost = 0
        val_n_correct = 0 
        val_total_l1 = 0
        val_all_labels = []
        val_all_predictions = []

        # for input, target in tqdm(train_loader):
        for batch in tqdm(val_loader):
            # input image and target is the class predicition
            v_input, v_target = batch['cine'], batch['target_AS']
            v_input = v_input.to(device, non_blocking=True)
            v_target = v_target.to(device, non_blocking=True)

            with torch.no_grad():
                v_output, v_min_distances, global_features, local_features = sprnet.image_embed(v_input)
                _, v_predicted = torch.max(v_output.data, 1)
                val_batch += 1

                ############ Compute Loss ###############

                val_cross_entropy = ce_loss.compute(v_output, v_target)

                val_cluster_cost = Cluster.compute(v_min_distances, v_target)
                
                val_separation_cost =  Separation.compute(v_min_distances, v_target)

                val_fc_lnorm = sprnet.last_layer.weight.norm(p=1)

                val_n_correct += (v_predicted == v_target).sum().item()

                val_total_cross_entropy += val_cross_entropy.item()
                val_total_cluster_cost += val_cluster_cost.item()
                val_total_separation_cost += val_separation_cost.item()
                val_total_l1 += val_fc_lnorm.item()

                v_loss = val_cross_entropy +  val_cluster_cost  + val_separation_cost + val_fc_lnorm
                val_loss += v_loss.item()

                val_n_examples += v_target.size(0)
                val_all_labels.extend(v_target.cpu().numpy())
                val_all_predictions.extend(v_predicted.cpu().numpy())

            del v_input
            del v_target
            del v_output
            del v_predicted
            del v_min_distances            


        val_total_loss = val_loss / val_batch
        val_total_cross_entropy  /=  val_batch
        val_total_cluster_cost /= val_batch
        val_total_separation_cost /=  val_batch
        val_total_l1 = val_total_l1/ val_batch
        Accuracy = (val_n_correct / n_examples * 100)
        val_balanced_acc = balanced_accuracy_score(val_all_labels, val_all_predictions)
        val_avg_f1 = f1_score(val_all_labels, val_all_predictions, average='macro')




        print(f"EPOCH {(epoch+1) }/{epochs} val loss: {val_total_loss}")

                # save the ckp based on the val loss

        
        # Save best model based on validation loss
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save({'epoch': epoch+1, 'model_state_dict': sprnet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_total_loss}, "trial/best_checkpoint.pth")
            

        
        wandb.log(
                    {
                    "Validation cross_entropy": val_total_cross_entropy,
                    "Validation cluster_cost": val_total_cluster_cost,
                    "Validation separation_cost": val_total_separation_cost,
                    "Validation fc_lnorm": val_total_l1,
                    "Balaced Acc":val_balanced_acc,
                    "f1_score":val_avg_f1,
                    "Total Validation Loss": val_total_loss,
                    "Epoch": epoch,
                }
            )


        # ---- Live Plotting ----
        ax.clear()
        ax.plot(epoch, val_total_cross_entropy, label="Validation cross_entropy", marker="o")
        ax.plot(epoch, val_total_cluster_cost, label="Validation cluster_cost", marker="s")
        ax.plot(epoch, val_total_separation_cost, label= "Validation separation_cost", marker="^")
        ax.plot(epoch, val_total_l1, label="Validation fc_lnorm", marker="d"),
        ax.plot(epoch, val_balanced_acc, label="Validation Balaced Acc", marker="d"),
        ax.plot(epoch, val_avg_f1, label="Validation f1_score", marker="d"),
        ax.plot(epoch, val_total_loss, label="Total Validation Loss", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("val Loss Curve")
        ax.legend()
        ax.grid(True)

        plt.pause(0.1)  # Small delay to update the figure




        



# Finalize plot
plt.ioff()
plt.show()
print('train completed. have a nice day!')
