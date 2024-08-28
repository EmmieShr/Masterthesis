import pandas as pd
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import f1_score, accuracy_score, classification_report

data1 = pd.read_csv('RealTrain.csv')  
data2 = pd.read_csv('MainTest.csv')

data1 = data1[['text', 'label']]
# data1.rename({'response': 'label'}, axis=1, inplace=True)

data2 = data2[['text', 'label']]


# Convert to Dataset
train_ds = Dataset.from_pandas(data1)
test_ds = Dataset.from_pandas(data2)

# Load SetFit model
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=8,
    num_iterations=20, 
    num_epochs=5,
    seed=18
)

# Train and evaluate!
trainer.train()
metric = trainer.evaluate()

# Save the model
model.save_pretrained('./Mainsetfitreal')
print('DONE')

# Make predictions on the test set
# y_true = test_ds["label"]
# y_pred = model.predict(test_ds["text"])

# f1 = f1_score(y_true, y_pred, average="weighted")
# accuracy = accuracy_score(y_true, y_pred)
# print(classification_report(y_true, y_pred))

# pred = pd.DataFrame({
#     'Text': test_ds['text'],
#     'Labels': test_ds['label'],
#     'Predicition': y_pred
# })

#pred.to_csv('sf_prediction_syn18_new.csv')
