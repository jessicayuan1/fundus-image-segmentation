import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from utils.metrics import calculate_iou_per_class, calculate_f1_per_class, calculate_recall_per_class
def valid_one_epoch(model, dataloader, criterion, device, n_classes=5):
    model.eval()
    val_loss = 0.0
    class_ious = [[] for _ in range(n_classes)]
    class_f1s = [[] for _ in range(n_classes)]
    class_recalls = [[] for _ in range(n_classes)]
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
            
            # Convert to binary predictions for metrics
            preds = torch.sigmoid(outputs)
            
            # Calculate metrics, can be disabled if only loss is desired
            batch_ious = calculate_iou_per_class(preds, masks, n_classes=n_classes)
            batch_f1s = calculate_f1_per_class(preds, masks, n_classes=n_classes)
            batch_recalls = calculate_recall_per_class(preds, masks, n_classes=n_classes)

            # Accumulate metrics
            for i in range(n_classes):
                if batch_ious[i] is not None:
                    class_ious[i].append(batch_ious[i])
                if batch_f1s[i] is not None:
                    class_f1s[i].append(batch_f1s[i])
                if batch_recalls[i] is not None:
                    class_recalls[i].append(batch_recalls[i])
    
    # Aggregate results
    epoch_loss = val_loss / len(dataloader.dataset)
    epoch_ious = [sum(cls) / len(cls) if cls else 0 for cls in class_ious]
    epoch_f1s = [sum(cls) / len(cls) if cls else 0 for cls in class_f1s]
    epoch_recalls = [sum(cls) / len(cls) if cls else 0 for cls in class_recalls]
    
    return epoch_loss, epoch_ious, epoch_f1s, epoch_recalls