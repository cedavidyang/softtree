# %%
import os
import torch
import numpy as np
import pandas as pd

from softtree.softtree_classification import SoftTreeClassifier
from softtree.oblique_tree import ParameterizedObliqueTree

# %%

if __name__ == "__main__":
    # input
    # model_name = "STC_lr2e-03_d6b100a1.000_lm0e+00le0e+00lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b1a1.000_lm0e+00le0e+00lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b0a1.000_lm0e+00le0e+00lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm0e+00le0e+00lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b0a1.096_lm0e+00le0e+00lgm0e+00.pt"
    # pruning_threshold = 0
    # model_name = "STC_lr2e-03_d6b1a1.047_lm1e-08le0e+00lgm0e+00.pt"
    model_name = "STC_lr2e-03_d6b1a1.047_lm1e-04le0e+00lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm1e-03le0e+00lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm1e-02le0e+00lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm0e+00le1e-08lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm0e+00le1e-04lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm0e+00le1e-02lgm0e+00.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm0e+00le0e+00lgm1e-08.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm0e+00le0e+00lgm1e-04.pt"
    # model_name = "STC_lr2e-03_d6b1a1.047_lm0e+00le0e+00lgm1e-02.pt"
    pruning_threshold = 4e-1
    lp_threshold = 1e-6
    model_path = os.path.join("./models", model_name)

    # load data
    imported_data = np.load('data/make_gaussian_10000_seed42.npz')
    X_train, y_train = imported_data['X_train'], imported_data['y_train']
    X_val, y_val = imported_data['X_val'], imported_data['y_val']
    X_test, y_test = imported_data['X_test'], imported_data['y_test']

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
    
    # create oblique tree
    weights = loaded_STC_model.inner_nodes.weight.detach().numpy()
    biases = loaded_STC_model.inner_nodes.bias.detach().numpy()
    leaf_logits = loaded_STC_model.leaf_nodes.leaf_scores.detach().numpy()
    leaf_values = np.argmax(leaf_logits, axis=1)

    odt_model = ParameterizedObliqueTree(
        tree_depth, weights, biases, leaf_values,
    )
    if pruning_threshold > 0:
        ratios = np.divide(
            weights,
            biases[:, np.newaxis],
            where=biases[:, np.newaxis] != 0
        )
        pruned_node_mask = np.all(np.abs(ratios) <= pruning_threshold, axis=1)
        pruned_weight_mask = np.abs(ratios) <= pruning_threshold
        weights[pruned_weight_mask] = 0.0

        odt_model.prune_zero_weight_branches()
        odt_model.prune_infeasible_paths(epsilon=lp_threshold)
        odt_model.prune_identical_leaves()

    odt_model.visualize()
    print(f"[*] Oblique tree created and pruned from {model_name}")
    
    train_accuracy = odt_model.score(X_train, y_train) 
    val_accuracy = odt_model.score(X_val, y_val) 
    test_accuracy = odt_model.score(X_test, y_test) 

    # save to file
    os.makedirs('./results', exist_ok=True)
    res_name = 'O' + model_name[1:-3] + f'_prune{pruning_threshold:.0e}.csv'
    res_path = os.path.join('./results', res_name)

    model_res = {
        'STC_model_name': model_name,
        'prune_threshold': pruning_threshold,
        'lp_threshold': lp_threshold,

        'pruned_nodes': np.sum(pruned_node_mask),
        'pruned_weights': np.sum(pruned_weight_mask),
        'internal_nodes': odt_model.internal_num,
        'leaf_nodes': odt_model.leaf_num,
        
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }

    model_res_df = pd.DataFrame.from_dict(model_res, orient='index', columns=['value'])
    model_res_df.to_csv(res_path)
    print(f"[*] Model performance saved successfully to {res_path}")
    
    # display
    print(f'Train Accuracy (converted oblique model): {train_accuracy:.4f}')
    print(f'Val Accuracy (converted oblique model): {val_accuracy:.4f}')
    print(f'Test Accuracy (converted oblique model): {test_accuracy:.4f}')

# %%