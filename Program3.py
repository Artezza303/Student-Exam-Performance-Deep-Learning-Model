import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#np.set_printoptions(linewidth=999999)


pd.set_option('expand_frame_repr', False)
#pd.set_option('display.max_columns', None)
#pd.set_option("max_rows", None)

df = pd.read_csv('StudentsPerformance.csv')



# OPTION TWO TO ONE HOT ENCODE
one_hot_encoded_data = pd.get_dummies(df, columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"])

one_hot_encoded_data = one_hot_encoded_data.astype('float32')


# OPTION TWO TO MOVE COLUMS
one_hot_encoded_data = one_hot_encoded_data[['gender_female', 'gender_male', 'race/ethnicity_group A', 'race/ethnicity_group B', 'race/ethnicity_group C',
        'race/ethnicity_group D', 'race/ethnicity_group E', 'parental level of education_associate\'s degree',
        'parental level of education_bachelor\'s degree', 'parental level of education_high school',
        'parental level of education_master\'s degree', 'parental level of education_some college',
        'parental level of education_some high school', 'lunch_free/reduced', 'lunch_standard',
        'test preparation course_completed', 'test preparation course_none','math score', 'reading score', 'writing score']]

#print(one_hot_encoded_data)


#Min Max Normalize
#one_hot_encoded_data = (one_hot_encoded_data-one_hot_encoded_data.min())/(one_hot_encoded_data.max()-one_hot_encoded_data.min())



#print(one_hot_encoded_data)





numpy_dataset = one_hot_encoded_data.values
x_train, x_test, y_train, y_test = train_test_split(numpy_dataset[:,:17],numpy_dataset[:,17:],test_size=0.15)



math_score_train, reading_score_train, writing_score_train = y_train[:,0], y_train[:,1], y_train[:,2]
math_score_test, reading_score_test, writing_score_test = y_test[:,0], y_test[:,1], y_test[:,2]

output_train = [math_score_train, reading_score_train, writing_score_train]
output_test = [math_score_test, reading_score_test, writing_score_test]



input_layer=tf.keras.layers.Input(shape=(17,))

#default 128, 21
first_dense=tf.keras.layers.Dense(24,activation='relu')(input_layer)
math_output=tf.keras.layers.Dense(1,name='math_prediction')(first_dense)


second_dense=tf.keras.layers.Dense(24,activation='relu')(first_dense)
reading_output=tf.keras.layers.Dense(1,name='reading_prediction')(second_dense)


third_dense=tf.keras.layers.Dense(24,activation='relu')(second_dense)
writing_output=tf.keras.layers.Dense(1,name='writing_prediction')(third_dense)


#output_list = [math_output, reading_output, writing_output]
model=tf.keras.Model(inputs = input_layer, outputs = [math_output,reading_output,writing_output])


#optimzers = tf.keras.optimizers.SGD(learning_rate = 0.001)
optimzers = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
#optimzers = tf.keras.optimizers.Adam(learning_rate = 0.001)


#VERSION 2
model.compile(optimizer=optimzers,
              loss={'math_prediction' : 'mse' ,'reading_prediction' : 'mse','writing_prediction' : 'mse'},
              metrics={'math_prediction' : tf.keras.metrics.MeanSquaredError() ,'reading_prediction' : tf.keras.metrics.MeanSquaredError(),'writing_prediction' : tf.keras.metrics.MeanSquaredError()})


#output_train = [math_score_train, reading_score_train, writing_score_train]
output_validation = [math_score_test, reading_score_test, writing_score_test]



history = model.fit(x_train,output_train, epochs = 35, validation_data=(x_test,output_validation))







# Test the model and print loss and rmse for both outputs
loss, Y1_loss, Y2_loss, Y3_loss, Y1_rmse, Y2_rmse, Y3_rmse = model.evaluate(x=x_test, y=y_test)

print()
print(f'loss: {loss}')
print(f'math_loss: {Y1_loss}')
print(f'math_rmse: {Y1_rmse}')

print(f'reading_loss: {Y2_loss}')
print(f'reading_rmse: {Y2_rmse}')

print(f'writing_loss: {Y3_loss}')
print(f'writing_rmse: {Y3_rmse}')







def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()



Y_pred = model.predict(x_test)
math_pred = Y_pred[0]
reading_pred = Y_pred[1]
writing_pred = Y_pred[2]


correct_math = 0.0;
correct_reading = 0.0;
correct_writing = 0.0;


for f, b in zip(math_score_test, Y_pred[0]):
    print('True: ', f, ' Prediction: ' , b, ' Difference: ', abs(f-b))
    if(abs(f-b) <= 10):
        correct_math += 1.0

print(' ')

for f, b in zip(reading_score_test, Y_pred[1]):
    print('True: ', f, ' Prediction: ' , b, ' Difference: ', abs(f-b))
    if(abs(f-b) <= 10):
        correct_reading += 1.0
    
print(' ')

for f, b in zip(writing_score_test, Y_pred[2]):
    print('True: ', f, ' Prediction: ' , b, ' Difference: ', abs(f-b))
    if(abs(f-b) <= 10):
        correct_writing += 1.0

print('Math  Acc: ', correct_math / 150)
print('Read  Acc: ', correct_reading / 150)
print('Write Acc: ', correct_writing / 150)
print(' ')




#ActualMath = math_score_test
#PredictMath = Y_pred[0]

#ActualReading = reading_score_test
#PredictReading = Y_pred[1]

#ActualWriting = writing_score_test
#PredictWriting = Y_pred[2]



#mapeObject = keras.losses.MeanAbsolutePercentageError()


#mapeMath = mapeObject(ActualMath, PredictMath)
#mapeReading = mapeObject(ActualReading, PredictReading)
#mapeWriting = mapeObject(ActualWriting, PredictWriting)


#mape_score_math = mapeMath.numpy()
#mape_score_reading = mapeReading.numpy()
#mape_score_writing = mapeWriting.numpy()

 
#print(mape_score_math)
#print(mape_score_reading)
#print(mape_score_writing)





plot_diff(math_score_test, Y_pred[0], title='Math')
plot_diff(reading_score_test, Y_pred[1], title='Reading')
plot_diff(writing_score_test, Y_pred[2], title='Writing')


plot_metrics(metric_name='math_prediction_root_mean_squared_error', title='Math RMSE', ylim=80)
plot_metrics(metric_name='reading_prediction_root_mean_squared_error', title='Reading RMSE', ylim=80)
plot_metrics(metric_name='writing_prediction_root_mean_squared_error', title='Writing RMSE', ylim=80)


model.save('./model_student_performance/', save_format='tf')




















