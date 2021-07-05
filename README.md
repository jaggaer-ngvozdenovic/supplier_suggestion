# supplier_suggestion

Two main scripts are `comress_features.ipynb` and `main.ipynb`.

With `compress_features.ipynb` you can compress and save features and tagrget variable in `.tfrecord` files.

`Main.ipynb` contains training, prediction and evaluation for the problem you want to solve 
(0 = classification, 1 = binary classification, 2 = regression).
It uses compressed features that are saved with `compress_features.ipynb` and extracts them in desirable format.
