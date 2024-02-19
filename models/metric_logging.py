import pandas as pd
import os


metric_file_name = 'Metric.xlsx'
columns = ['Model','Run','Train Accuracy','Train Loss','Val Accuracy','Val Loss','Best Val Loss', \
           'Best Val Loss Path','Best Val Accuracy','Best Val Accuracy Path','Test Loss','Test Accuracy','Top 1 Train','Top 1 Val Accuracy','Top 1 Test Accuracy']

def get_df(run_dir):
    if not os.path.isfile(metric_file_name):
        df = pd.DataFrame(columns=columns)
        df.to_excel(metric_file_name, index=False)
    else:
        df = pd.read_excel(metric_file_name)
    return df

def add_to_database(df,model_name, run, train_accuracy, train_loss, val_accuracy, val_loss, \
                    best_val_loss, best_val_loss_path, best_val_accuracy, best_val_accuracy_path,test_loss,test_metric,top1_train_accuracy, top1_val_accuracy,top1_test_accuracy):
    
    new_row = {
        'Model': model_name,
        'Run': run,
        'Train Accuracy': train_accuracy,
        'Train Loss': train_loss,
        'Val Accuracy': val_accuracy,
        'Val Loss': val_loss,
        'Best Val Loss': best_val_loss,
        'Best Val Loss Path': best_val_loss_path,
        'Best Val Accuracy': best_val_accuracy,
        'Best Val Accuracy Path': best_val_accuracy_path,
        'Test Loss' : test_loss,
        'Test Accuracy' : test_metric,
        'Top 1 Train' : top1_train_accuracy, 
        'Top 1 Val Accuracy' : top1_val_accuracy,
        'Top 1 Test Accuracy' : top1_test_accuracy

    }
    # df = df.append(new_row, ignore_index=True)
    df.loc[len(df)] = new_row
    df.to_excel(metric_file_name, index=False)
