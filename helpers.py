
"""
Most of these functions were taken & adapted from the course materials.
"""

import numpy as np
import pandas as pd
import networkx as nx
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, roc_auc_score
from pyBKT.models import Model

def get_difficulty(doc):
    res = None
    doc = json.loads(doc)
    if doc.get('metaData') != None:
        res = doc.get('metaData').get('estimatedDifficulty')
    return res

def create_iterator(data, n_splits=5):
    """ Creates an iterator to split interactions in data into train and test,
    with the same student not appearing in two different folds.

    Params:
        data (pd.DataFrame):        Dataframe with student's interactions.
    Returns:
        generator: Iterator over the train/test splits.
    """
    X = np.arange(len(data.index))
    groups = data['user_id'].values
    group_shuffle = GroupShuffleSplit(n_splits=n_splits, train_size=.8, test_size=0.2, random_state=0)
    return group_shuffle.split(X, groups=groups)

def train_bkt(df, defaults = {'skill_name': 'topic_id', 'correct': 'evaluation'}):
    """ Trains the BKT model and makes predictions on the test set.

    Params:
        df (pd.DataFrame): Input data.
        defaults (dict): Mapping of columns to the format the BKT package expects.
    Returns:
        Trained BKT model.
    """
    rmse_bkt, auc_bkt = [], []
    for it, (train_index, test_index) in enumerate(create_iterator(df)):
        # Split data in training and test sets
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        # Initialize and fit the model
        model = Model(seed=0)
        model.fit(data=X_train, defaults=defaults)
        # Compute RMSE
        train_rmse = model.evaluate(data=X_train, metric='rmse')
        test_rmse = model.evaluate(data=X_test, metric='rmse')
        rmse_bkt.append(test_rmse)
        # Compute AUC
        train_auc = model.evaluate(data=X_train, metric='auc')
        test_auc = model.evaluate(data=X_test, metric='auc')
        auc_bkt.append(test_auc)

        print('Split', it, ': RMSE = {:.3f}'.format(train_rmse, test_rmse),'AUC = {:.3f}'.format(train_auc, test_auc))

    print('\nMean RMSE (std): {:.3f} ({:.3f})'.format(np.mean(rmse_bkt), np.std(rmse_bkt)))
    print('Mean AUC (std): {:.3f}, ({:.3f})'.format(np.mean(auc_bkt), np.std(auc_bkt)))

    return model,rmse_bkt, auc_bkt

def avg_y_by_x(x, y):
    '''
    Compute average learning curve and number of students over the number of opportunities.
    x is the number of opportunities.
    y the success rates of the users (can be predicted success rate or true success rate).
    '''
    # Transform lists into arrays
    x = np.array(x)
    y = np.array(y)

    # Sort the integer id representing the number of opportunities in increasing order
    xs = sorted(list(set(x)))

    # Supporting lists to store the:
    # - xv: integer identifier of the number of opportunities
    # - yv: average value across students at that number of opportunities
    # - lcb and ucb: lower and upper confidence bound
    # - n_obs: number of observartions present at that number of opportunities (on per-skill plots, it is the #students)
    xv, yv, lcb, ucb, n_obs = [], [], [], [], []

    # For each integer identifier of the number of opportunities 0, ...
    for v in xs:
        ys = [y[i] for i, e in enumerate(x) if e == v] # We retrieve the values for that integer identifier
        if len(ys) > 0:
            xv.append(v) # Append the integer identifier of the number of opportunities
            yv.append(sum(ys) / len(ys)) # Append the average value across students at that number of opportunities
            n_obs.append(len(ys)) # Append the number of observartions present at that number of opportunities


            # Prepare data for confidence interval computation
            unique, counts = np.unique(ys, return_counts=True)
            counts = dict(zip(unique, counts))

            if 0 not in counts:
                counts[0] = 0
            if 1 not in counts:
                counts[1] = 0

            # Calculate the 95% confidence intervals
            ci = sc.stats.beta.interval(0.95, 0.5 + counts[0], 0.5 + counts[1])
            lcb.append(ci[0])
            ucb.append(ci[1])

    return xv, yv, lcb, ucb, n_obs


def plot_learning_curve(preds, title='Learning curve'):
    '''
    Plot learning curve using BKT model for a skill.
    '''
    xp = []
    yp = {}
    for col in preds.columns: # For y_true and and y_pred_bkt columns, initialize an empty list for curve values
        if 'y_' in col:
            yp[col] = []

    for user_id in preds['user_id'].unique(): # For each user
        user_preds = preds[preds['user_id'] == user_id] # Retrieve the predictions on the current skill for this user
        xp += list(np.arange(len(user_preds))) # The x-axis values go from 0 to |n_opportunities|-1
        for col in preds.columns:
            if 'y_' in col: # For y_true and and y_pred_bkt columns
                yp[col] += user_preds[col].tolist() # The y-axis value is the success rate for this user at that opportunity

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]}) # Initialize the plotting figure

    lines = []
    for col in preds.columns:
        if 'y_' in col: # For y_true and and y_pred_bkt columns
            x, y, lcb, ucb, n_obs = avg_y_by_x(xp, yp[col]) # Calculate mean and 95% confidence intervals for success rate
            y = [1-v for v in y] # Transform success rate in error rate
            if col == 'y_true': # In case of ground-truth data, we also show the confidence intervals
                axs[0].fill_between(x, lcb, ucb, alpha=.3)
            model_line, = axs[0].plot(x, y, label=col) # Plot the curve
            lines.append(model_line) # Store the line to then set the legend

    # Make decorations for the learning curve plot
    axs[0].legend(handles=lines)
    axs[0].set_ylabel('Error')
    axs[0].set_title(title)

    # Plot the number of observations per number of opportunities bars and make decorations
    axs[1].set_xlabel('# Questions answered')
    axs[1].bar([i for i in range(len(n_obs))], n_obs)
    axs[1].set_ylabel('# Observations')

    # Plot the learning curve and the bar plot
    plt.show()

def compute_metrics(df_preds, prediction_col_name):
    '''
    Compute RMSE and AUC based on the model prediction.
    '''
    overall_rmse = mean_squared_error(df_preds.evaluation, df_preds[prediction_col_name], squared=False)

    if 'prev_partial_attempts' in df_preds:
        eval_categorized = (df_preds.evaluation * 2).astype(int)
        pred_probs = pd.get_dummies(pd.cut(df_preds[prediction_col_name], [0, 1/3, 2/3, 1]))
        overall_auc = roc_auc_score(eval_categorized, pred_probs, multi_class='ovo')
    else:
        overall_auc = roc_auc_score(df_preds.evaluation, df_preds[prediction_col_name], multi_class='ovo')

    return overall_rmse, overall_auc

def compute_metrics_per_topic(df_preds, skill_col_name, prediction_col_name):
    '''
    Compute RMSE and AUC per topic based on the model prediction.
    '''
    metrics = df_preds.groupby(skill_col_name).apply(compute_metrics, prediction_col_name=prediction_col_name)
    metrics_df = pd.DataFrame(metrics.to_list(), columns=['rmse', 'auc'])
    metrics_mean, metrics_std = metrics_df.mean(axis=0), metrics_df.std(axis=0)
    return metrics_mean, metrics_std, metrics_df

def plot_metrics(rmse, rmse_per_topic, auc, auc_per_topic, topics):
    afm_rmse_df = pd.DataFrame([['Overall', rmse]], columns =['x', 'RMSE'])
    afm_rmse_per_topic_df = pd.DataFrame(list(zip(topics, rmse_per_topic)), columns =['x', 'RMSE'])

    afm_auc_df = pd.DataFrame([['Overall', auc]], columns =['x', 'AUC'])
    afm_auc_per_topic_df = pd.DataFrame(list(zip(topics, auc_per_topic)), columns =['x', 'AUC'])

    # RMSE
    # Overall RMSE
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    sns.barplot(ax=axes[0], x='x', y='RMSE', data=afm_rmse_df, estimator=np.mean, ci='sd')
    axes[0].set_title('Overall RMSE')
    axes[0].set_xlabel('')

    # RMSE per topic
    sns.barplot(ax=axes[1], x='x', y='RMSE', data=afm_rmse_per_topic_df, estimator=np.mean, ci='sd')
    plt.xticks(rotation=90)
    axes[1].set_title('RMSE across topics')
    axes[1].set_xlabel('')

    fig.show()

    # AUC
    # Overall AUC
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    sns.barplot(ax=axes[0], x='x', y='AUC', data=afm_auc_df, estimator=np.mean, ci='sd')
    axes[0].set_title('Overall AUC')
    axes[0].set_xlabel('')

    # AUC per topic
    sns.barplot(ax=axes[1], x='x', y='AUC', data=afm_auc_per_topic_df, estimator=np.mean, ci='sd')
    plt.xticks(rotation=90)
    axes[1].set_title('AUC across topics')
    axes[1].set_xlabel('')

    fig.show()


def prepare_seq(df):
    '''
    Extract user_id sequence in preparation for DKT. The output of this function
    feeds into the prepare_data() function.
    '''
    # Enumerate skill id as a categorical variable
    # (i.e. [32, 12, 32, 45] -> [0, 1, 0, 2])

    df['skill'], skill_codes = pd.factorize(df['topic_id'], sort=True)

    n_classes = df['evaluation'].nunique()

    # Cross skill id with answer to form a synthetic feature
    df['skill_with_answer'] = (df['skill'] * n_classes + df['evaluation'] * (n_classes + 1)//2).astype(int)

    # df['difficulty'] = df['difficulty'].astype(int)

    # Convert to a sequence per user_id and shift features 1 timestep
    seq = df.groupby('user_id').apply(lambda r: (r['skill_with_answer'].values[:-1],
                                                 r['skill'].values[1:],
                                                 r['evaluation'].values[1:],
                                                 r['difficulty'].values[1:], ))

    # Get max skill depth and max feature depth
    skill_depth = df['skill'].max()
    features_depth = df['skill_with_answer'].max() + 1

    return seq, features_depth, skill_depth

def prepare_data(seq, params, features_depth, skill_depth, incl_difficulty=False):
    '''
    Manipulate the data sequences into the right format for DKT with padding by batch
    and encoding categorical features.
    '''

    # Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(generator=lambda: seq, output_types=(tf.int32, tf.int32, tf.float32, tf.float32))

    # Encode categorical features and merge skills with labels to compute target loss
    if incl_difficulty:
        dataset = dataset.map(
            lambda feat, skill, label, difficulty: (
                tf.concat(values=[tf.one_hot(feat, depth=features_depth), tf.expand_dims(difficulty, -1)], axis=-1),
                tf.concat(values=[tf.one_hot(skill, depth=skill_depth), tf.expand_dims(label, -1)], axis=-1)
            )
        )
    else:
        dataset = dataset.map(
        lambda feat, skill, label, _: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(values=[tf.one_hot(skill, depth=skill_depth), tf.expand_dims(label, -1)], axis=-1)
        )
)

    # Pad sequences to the appropriate length per batch
    dataset = dataset.padded_batch(
        batch_size=params['batch_size'],
        padding_values=(params['mask_value'], params['mask_value']),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    return dataset.repeat(), len(seq)

def create_model(nb_features, nb_skills, params):

    # Create an LSTM model architecture
    inputs = tf.keras.Input(shape=(None, nb_features), name='inputs')

    # We use a masking layer here to ignore our masked padding values
    x = tf.keras.layers.Masking(mask_value=params['mask_value'])(inputs)

    # This LSTM layer is the crux of the model; we use our parameters to specify
    # what this layer should look like (# of recurrent_units, fraction of dropout).
    x = params['backbone_nn'](params['recurrent_units'], return_sequences=True, dropout=params['dropout_rate'])(x)

    # We use a dense layer with the sigmoid function activation to map our predictions
    # between 0 and 1.
    dense = tf.keras.layers.Dense(nb_skills, activation=params['activation'])

    # The TimeDistributed layer takes the dense layer predictions and applies the sigmoid
    # activation function to all time steps.
    outputs = tf.keras.layers.TimeDistributed(dense, name='outputs')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='DKT')

    # Compile the model with our loss functions, optimizer, and metrics.
    model.compile(loss=params['loss_fn'],
                  optimizer=params['optimizer'],
                  metrics=params['metrics'])

    return model

class AUC(tf.keras.metrics.AUC):
    # Our custom AUC calls our get_target function first to remove predictions on padded values,
    # then computes a standard AUC metric.
    def __init__(self):
        # We use a super constructor here just to make our metric name pretty!
        super(AUC, self).__init__(name='auc')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(AUC, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)

class MultiAUC(tf.keras.metrics.AUC):
    def __init__(self):
        super(MultiAUC, self).__init__(name='multi-auc', multi_label=True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        pred = tf.reshape(pred, [-1])
        true = tf.reshape(true, [-1])
        pred_thres = tf.floor(pred*3)/2
        true = tf.one_hot(tf.cast(2*true, tf.int32), depth=3, axis=1)
        pred = tf.one_hot(tf.cast(2*pred_thres, tf.int32), depth=3, axis=1)
        super(MultiAUC, self).update_state(y_true=true, y_pred=pred)

class RMSE(tf.keras.metrics.RootMeanSquaredError):
    # Our custom RMSE calls our get_target function first to remove predictions on padded values,
    # then computes a standard RMSE metric.
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(RMSE, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)

class Precision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(Precision, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)

class Recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(Recall, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)

def CustomBinaryCrossEntropy(y_true, y_pred):
    # Our custom binary cross entropy loss calls our get_target function first
    # to remove predictions on padded values, then computes standard binary cross-entropy.
    y_true, y_pred = get_target(y_true, y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def CustomMeanSquaredError(y_true, y_pred):
    y_true, y_pred = get_target(y_true, y_pred)
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def get_target(y_true, y_pred, mask_value=-1):
    '''
    Adjust y_true and y_pred to ignore predictions made using padded values.
    '''
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, mask_value), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred
