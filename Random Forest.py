import numpy as np
import pandas as pd
import os
import librosa
import scipy
from scipy.stats import skew
from tqdm import tqdm, tqdm_pandas
import matplotlib.pyplot as plt
tqdm.pandas()

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 44100
audio_train_files = os.listdir('/Users/Aiswaryaa/all/audio_musical_train')
#audio_test_files = os.listdir('../input/audio_test')

train = pd.read_csv('/Users/Aiswaryaa/all/train_music.csv')

def clean_filename(fname, string):   
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':        
        file_name = string + file_name
    return file_name

# Generate mfcc features with mean and standard deviation
def get_mfcc(name, path):
    data, _ = librosa.core.load(path + name, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        ft1 = librosa.feature.mfcc(data, sr = SAMPLE_RATE, n_mfcc=30)
        ft2 = librosa.feature.zero_crossing_rate(data)[0]
        ft3 = librosa.feature.spectral_rolloff(data)[0]
        ft4 = librosa.feature.spectral_centroid(data)[0]
        ft5 = librosa.feature.spectral_contrast(data)[0]
        ft6 = librosa.feature.spectral_bandwidth(data)[0]
        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
    except:
        print('bad file')
        return pd.Series([0]*210)
        
def convert_to_labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids

train_data = pd.DataFrame()
train_data['fname'] = train['fname']
#test_data = pd.DataFrame()
#test_data['fname'] = audio_test_files

train_data = train_data['fname'].progress_apply(get_mfcc, path='/Users/Aiswaryaa/all/audio_musical_train/')
print('done loading train mfcc')
#test_data = test_data['fname'].progress_apply(get_mfcc, path='../input/audio_test/')
#print('done loading test mfcc')

train_data['fname'] = train['fname']
#test_data['fname'] = audio_test_files

train_data['label'] = train['label']
#test_data['label'] = np.zeros((len(audio_test_files)))

train_data.head()
X = train_data.drop(['label', 'fname'], axis=1)
feature_names = list(X.columns)
X = X.values
labels = np.sort(np.unique(train_data.label.values))
num_class = len(labels)
c2i = {}
i2c = {}
for i, c in enumerate(labels):
    c2i[c] = i
    i2c[i] = c
y = np.array([c2i[x] for x in train_data.label.values])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=65).fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(sum(pca.explained_variance_ratio_))

X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size = 0.2, random_state = 42)
rfc = RandomForestClassifier(n_estimators = 150)
rfc.fit(X_train, y_train)
print(rfc.score(X_val, y_val))
print(confusion_matrix(rfc.predict(X_val), y_val))

cm=confusion_matrix(rfc.predict(X_val), y_val)

p=list()
for i in range(len(cm)):
    d=0
    for j in range(len(cm[i])):
        d=d+ cm[j][i]
    p.append(cm[i][i]/d)
r=list()
for i in range(len(cm)):
    d=0
    for j in range(len(cm[i])):
        d=d+ cm[i][j]
    r.append(cm[i][i]/d)
    


b=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
plt.plot(b, p, 'b', label='Split Fraction')
plt.title('Precision curve')
plt.xlabel('Class')
plt.ylabel('Precision')
plt.legend()
plt.show()
plt.plot(b, r, 'b', label='Split Fraction')
plt.title('Recall curve')
plt.xlabel('Class')
plt.ylabel('Recall')
plt.legend()
plt.show()
f=list()
for i in range(len(r)):
    f.append(2*p[i]*r[i]/(p[i]+r[i]))
 plt.plot(b, f, 'b', label='Split Fraction')
plt.title('f1 score curve')
plt.xlabel('Class')
plt.ylabel('f1 score')
plt.legend()
plt.show()
avgp=0
c=0
for i in range(len(p)):
    c=c+p[i]
avgp=(c/17)
print("avg precision",avgp)
avgr=0
c=0
for i in range(len(r)):
    c=c+r[i]
avgr=(c/17)
print("avg Recall",avgr)

avgf=0
c=0
for i in range(len(f)):
    c=c+f[i]
avgf=(c/17)
print("avg F1 score",avgf)
