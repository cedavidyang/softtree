# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import torch

from softtree.training_util import STC_fit_NLLLoss_acc
from softtree.softtree_classification import SoftTreeClassifier

# %%
if __name__ == '__main__':
    # random seed
    torch.manual_seed(404)

    # hyperparameters
    input_size = 2
    num_classes = 4

    tree_depth = 6
    # beta, beta_epoch, beta_anneal = 100, 1, 1.0
    # beta, beta_epoch, beta_anneal = 1.0, 1, 1.0
    # beta, beta_epoch, beta_anneal = 0.01, 1, 1.0
    beta, beta_epoch, beta_anneal = 1.0, 1, 100**(1/100)
    # beta, beta_epoch, beta_anneal = 0.01, 1, (100/0.01)**(1/100)

    batch_size = 32
    num_epochs = 100
    learning_rate = 0.002
    lr_epoch, lr_decay = 1, 1.0
    lmd_l1, lmd_l2, lmd_gl1 = 1e-3, 0.0, 0.0

    # load data
    # X = features (coordinates), y = labels (0, 1, 2, 3)
    imported_data = np.load('data/make_gaussian_10000_seed42.npz')
    X_train, y_train = imported_data['X_train'], imported_data['y_train']
    X_val, y_val = imported_data['X_val'], imported_data['y_val']
    X_test, y_test = imported_data['X_test'], imported_data['y_test']

    # prepare pytorch data
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # implement training
    STC_model, test_accuracy, train_loss_history, train_acc_history, \
        val_loss_history, val_acc_history = STC_fit_NLLLoss_acc(
            X_train_tensor, y_train_tensor,
            X_test_tensor, y_test_tensor,
            input_size, num_classes,
            tree_depth=tree_depth,
            beta=beta, beta_epoch=beta_epoch, beta_anneal=beta_anneal,
            batch_size=batch_size, num_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_epoch=lr_epoch, lr_decay=lr_decay,
            print_every=10,
            holdout_val=True, X_val_tensor=X_val_tensor, y_val_tensor=y_val_tensor,
            lambda_l1=lmd_l1,
            lambda_l2=lmd_l2,
            lambda_groupl1=lmd_gl1
    )

    # save model
    os.makedirs("./models", exist_ok=True)
    model_name = f"STC_lr{learning_rate:.0e}_d{tree_depth:d}b{beta:.0f}a{beta_anneal:.3f}_lm{lmd_l1:.0e}le{lmd_l2:.0e}lgm{lmd_gl1:.0e}.pt"
    model_path = os.path.join("./models", model_name)

    model_state = STC_model.state_dict()
    model_hyperparams = {
        'input_size': input_size,
        'num_classes': num_classes,
        'tree_depth': tree_depth,

        'beta': beta,
        'beta_epoch': beta_epoch,
        'beta_anneal': beta_anneal,

        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'lr_epoch': lr_epoch,
        'lr_decay': lr_decay,

        'lmd_l1': lmd_l1,
        'lmd_l2': lmd_l2,
        'lmd_gl1': lmd_gl1,
    }
    save_dict = {
        'model_state': model_state,
        'model_hyperparams': model_hyperparams,
    }

    torch.save(save_dict, model_path)
    print(f"[*] Classifier saved successfully to {model_path}")

    # save model performance
    os.makedirs('./results', exist_ok=True)
    res_name = f"STC_lr{learning_rate:.0e}_d{tree_depth:d}b{beta:.0f}a{beta_anneal:.3f}_lm{lmd_l1:.0e}le{lmd_l2:.0e}lgm{lmd_gl1:.0e}.xlsx"
    res_path = os.path.join('./results', res_name)

    # save train and val history to pandas dataframe
    train_log = defaultdict(list)
    train_log['batch'] = np.arange(len(train_loss_history))
    train_log['train_loss'] = train_loss_history
    train_log['train_acc'] = train_acc_history
    train_log['val_loss'] = val_loss_history
    train_log['val_acc'] = val_acc_history

    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_excel(res_path, sheet_name='train_log')

    model_hyperparams['test_accuracy'] = test_accuracy
    # model_info = pd.DataFrame(model_hyperparams, index=[0])
    model_info = pd.DataFrame.from_dict(model_hyperparams, orient='index', columns=['value'])

    # save model_info to a new spreadsheet called model_info
    with pd.ExcelWriter(res_path) as writer:
        model_info.to_excel(writer, sheet_name='model_info', index=True)
        train_log_df.to_excel(writer, sheet_name='train_log', index=False)

    print(f"[*] Model performance saved successfully to {res_path}")

    # temporary plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_loss_history, label='Train Loss')
    ax1.plot(val_loss_history, label='Validation Loss', color='orange')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_acc_history, label='Train Accuracy')
    ax2.plot(val_acc_history, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

# %%
# test loaded model

def _test_loaded_model():
    # input
    model_path = "./models/STC_d6b1a1.000_lm0e00le0e00lgm0e00.pt"

    # load data
    # X = features (coordinates), y = labels (0, 1, 2, 3)
    imported_data = np.load('data/make_gaussian_10000_seed42.npz')
    X_test, y_test = imported_data['X_test'], imported_data['y_test']
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # load model
    load_dict = torch.load(model_path)
    model_state = load_dict['model_state']
    model_hypers = load_dict['model_hyperparams']

    input_size = model_hypers['input_size']
    num_classes = model_hypers['num_classes']
    tree_depth = model_hypers['tree_depth']
    beta = model_hypers['beta']

    loaded_STC_model = SoftTreeClassifier(
        input_dim=input_size,
        output_dim=num_classes,
        depth=tree_depth,
        beta=beta,
        apply_batchNorm=False,
    )

    loaded_STC_model.load_state_dict(model_state)
    loaded_STC_model.eval()

    with torch.no_grad():
        prob_test = loaded_STC_model(X_test_tensor)
        _, y_test_pred = torch.max(prob_test.data, 1)
        accuracy = (y_test_pred == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Test Accuracy (loaded model): {accuracy:.4f}')