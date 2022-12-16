# createSpec.py
import autograd.numpy as np
import os
from seldonian.parse_tree.parse_tree import (ParseTree,
    make_parse_trees_from_constraints)

from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (load_json,save_pickle)
from seldonian.spec import SupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel) 
from seldonian.models import objectives

if __name__ == '__main__':
    data_pth = "new_dataset.csv"
    metadata_pth = "metadata.json"
    save_dir = '.'
    os.makedirs(save_dir,exist_ok=True)
    # Create dataset from data and metadata file
    regime='supervised_learning'
    sub_regime='classification'

    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')
    sensitive_col_names = dataset.meta_information['sensitive_col_names']

    # Use logistic regression model
    model = LogisticRegressionModel()
    
    # Set the primary objective to be log loss
    primary_objective = objectives.binary_logistic_loss
    
    # Define behavioral constraints 
    constraint_strs = ['FNR <= 0.2', 'FPR <= 0.5'] #
    deltas = [0.05, 0.05]
    
    # For each constraint (in this case only one), make a parse tree
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime,columns=sensitive_col_names)

    # Save spec object, using defaults where necessary
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime=sub_regime,
        frac_data_in_safety=0.6,
        primary_objective=primary_objective,
        initial_solution_fn=model.fit,
        use_builtin_primary_gradient_fn=True,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.01, 0.01]),
            'alpha_theta'   : 0.003,
            'alpha_lamb'    : 0.001,
            'beta_velocity' : 0.2,
            'beta_rmsprop'  : 0.2,
            'batch_size'    : 4000,
            'n_epochs'      : 2000,
            'use_batches'   : True,
            'num_iters'     : 3,
            'gradient_library': "autograd",
            'hyper_search'  : True,
            'verbose'       : True,
        }
    )

    spec_save_name = os.path.join(save_dir,'spec.pkl')
    save_pickle(spec_save_name,spec)
    print(f"Saved Spec object to: {spec_save_name}")