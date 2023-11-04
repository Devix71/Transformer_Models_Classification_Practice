# Transformer_Models_Classification_Practice

Within this repository, you can find my attempts to experiment, play, and alter the ViT transformer model.

Initially, I started by following the Huggingface🤗 tutorial on [image classification](https://huggingface.co/docs/transformers/tasks/image_classification). This tutorial focuses on creating a [Vision Transformer (ViT) model](https://huggingface.co/docs/transformers/model_doc/vit) with the express purpose of correctly classifying specified objects within the [Food-101 Huggingface dataset](https://huggingface.co/datasets/food101).

The notebook containing this attempt is called `Assignment2_CatsandHorses_Original.ipynb`.

After successfully implementing and fine-tuning the model to the Food-101 dataset, the model output the results:

```json
[
  {'score': 0.13332407176494598, 'label': 'beignets'},
  {'score': 0.02000979147851467, 'label': 'prime_rib'},
  {'score': 0.01762833632528782, 'label': 'chicken_wings'},
  {'score': 0.017359808087348938, 'label': 'bruschetta'},
  {'score': 0.014923209324479103, 'label': 'hamburger'}
]
```

This shows that when given the testing image, the model believes that it contains a "beignets" with 13% probability. This may not seem like much at first glance, but when comparing with the 2nd most confident category of "prime_rib", which has just 2% confidence, it looks like the model is fairly certain (and correctly so) that the object in the image is a "beignets".

## Adapting to the COCO Dataset

Moving on, I have decided to adapt the ViT model to a binary-classification problem by fine-tuning it on the [Common Objects in Context (COCO) dataset](https://cocodataset.org/). To achieve this, I had to adapt the data preprocessing and dataloading parts of the original notebook. I achieved this by first identifying and selecting the images exhibiting the categories I was interested in classifying (cats and horses) from the COCO dataset. Once that was done, I split the data into training, testing, and validating sub-datasets based on an 80-10-10 ratio with 2640 total images in each category to avoid overfitting and to achieve as close to real-world performance as possible.

The notebook containing the code is `Assignment2_CatsandHorses_COCO.ipynb`

After encoding the data and training the model on it, the end results were:
```json
[{'score': 0.9908796548843384, 'label': 'horse'},
{'score': 0.009120305068790913, 'label': 'cat'}]
```

Which shows a staggering overfitting on the 'horse' category, even if both the 'horse' and 'cat' categories were equally represented in the datasets. Not only that, but the object in the image belongs to the 'cat' category. The abysmal result may be a result of its training process or perhaps due to the small dataset (less than 3000 images for each category).

## Implementing Multiclass Labeling

Next, I tried to adapt the ViT model to do multiclass classification on the COCO dataset. This involved altering the data preprocessing steps by including more animal categories into the final datasets. Doing so led to encountering a challenge, namely the different numbers of images containing said objects. While the 'horse', 'cat', 'bird', and 'dog' had the necessary amount of images, the 'bear' and 'sheep' categories had around half or less than 2640 images. This shouldn't be a problem if the proportions are similar in the evaluation dataset, which they are due to the aforementioned splitting method.

The code can be found in the notebook `Assignment2_CatsandHorses_COCO-Multiclass.ipynb`

After running the model on the new dataset, the results are:

```json
[
    {'score': 0.37481778860092163, 'label': 'bear'},
    {'score': 0.21255673468112946, 'label': 'bird'},
    {'score': 0.18375423550605774, 'label': 'cat'},
    {'score': 0.1049196794629097, 'label': 'dog'},
    {'score': 0.06941524893045425, 'label': 'horse'}
]
```


Compared to the previous binary classifier, the multiclass model shows slightly better performance while still misclassifying the object in the image as "bear." This is a surprising result, as it seems the model has placed a larger weight on the 'bear' category with a 37% confidence. Somehow the true label 'cat' is the third most probable choice with 18% confidence.

I tried rechecking this with a classic CNN model to perform multiclass classification on the same dataset. This involved changing the entire data preprocessing stage, where the data was selected for the aforementioned categories and split according to the 80-10-10 ratio. Next the model was adapted to deal with the new tensor dimensions in its 2 layers, both by hardcoding the padding to 1 and by hardcoding the output of the layer to 30000 features. Not only that but the loss function was changed to a cross-entropy loss as this is a multiclass classification model. The evaluation functions were replaced to reflect the new output, thus the evaluation output being a classification report and a confusion matrix.

The results weren't relevant:

The code can be checked here `Demo 1.2 - cats and horses - CNN-Multiclass.ipynb`

### Classification Report

           precision    recall  f1-score   support

     cat       0.22      0.76      0.34       500
   horse       0.19      0.24      0.21       500
     dog       0.00      0.00      0.00       440
   sheep       0.00      0.00      0.00       306
    bird       0.00      0.00      0.00       440
    bear       0.00      0.00      0.00       192

accuracy                           0.21      2378
macro avg      0.07      0.17      0.09      2378
weighted avg   0.09      0.21      0.12      2378


### Confusion Matrix

       cat    horse  dog  sheep  bird  bear
cat    381     119   0     0     0       0
horse  380     120   0     0     0       0
dog    322     118   0     0     0       0 
sheep  233      73   0     0     0       0
bird   305     135   0     0     0       0
bear   124      68   0     0     0       0


This data suggests that the model only took into account the 'cat' and 'horse' objects for some reason.

## Conclusion

Overall, my attempt to correctly adapt the models to the COCO dataset for both binary and multiclass classification has been an astounding failure. But this isn't a reason to give up as I have learned many things through these challenges and results.


## Sidenote - Sequence Classification

I have also played around with a more NLP oriented model, namely the Huggingface🤗 [Token classification](https://huggingface.co/docs/transformers/tasks/token_classification). This model uses the [DistilBERT](https://huggingface.co/distilbert-base-uncased) model in classifiying tokens, fine-tuned on the [WNUT 17](https://huggingface.co/datasets/wnut_17) dataset.

The notebook containing the experiment is `BERT Word Sequence Classification Tutorial.ipynb`

After fine-tuning the model, it has been tested on the phrase `The Golden State Warriors are an American professional basketball team based in San Francisco.`, outputting these results:
```json
[{'entity': 'B-location',
  'score': 0.42658594,
  'index': 2,
  'word': 'golden',
  'start': 4,
  'end': 10},
 {'entity': 'I-location',
  'score': 0.35856345,
  'index': 3,
  'word': 'state',
  'start': 11,
  'end': 16},
 {'entity': 'B-group',
  'score': 0.30640018,
  'index': 4,
  'word': 'warriors',
  'start': 17,
  'end': 25},
 {'entity': 'B-location',
  'score': 0.6552351,
  'index': 13,
  'word': 'san',
  'start': 80,
  'end': 83},
 {'entity': 'B-location',
  'score': 0.4668664,
  'index': 14,
  'word': 'francisco',
  'start': 84,
  'end': 93}]
```
  Which clearly show that the model correctly identified the locations and the group being refferenced within the phrase.
