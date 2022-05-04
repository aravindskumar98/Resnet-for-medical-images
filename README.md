

* For a single model with 8Go GPU: ```python main.py```.

* With a 2Go GPU : python main.py --batch_split 4

To do 10 estimates with 8Go GPU: ```python main10.py```. After the learning, the following script will compute the mean prediction of the models on test dataset: ```python aggregate.py```.



