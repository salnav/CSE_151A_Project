# CSE_151A_Project

## Dataset Details

The folder with the dataset is linked [here](https://drive.google.com/drive/folders/1YwGAiLkXr5lPY1PL1VM97KHVrMha24rm?usp=sharing).
The original dataset and its details can be found [here](https://archive.ics.uci.edu/dataset/417/character+font+images) via the UC Irvine Machine Learning Repository from Richard Lyman.

## [Data Exploration Jupyter Notebook](CSE_151_Project_Data_Exploration.ipynb)

## Data Preprocessing

### I plan to...
- Scale the current image pixel values (currently 0-255) to 0-1, so the model can process the data better.
- Get rid of all the irrelevant features such as the font, fontVariant, strength, originalH, etc.
- Get rid of any missing data (although from my data exploration it seems are though there is no missing data).
- Split the data into a 80-10-10 train,test,validation split (respectively).
- Balance the fonts, so no one font is overrepresnted in the dataset.
- Subsample if the dataset is too large for the training step.
- Use binary encoding for the m_label class due to the large number of labels and the current ordinal numbers not being ideal.
