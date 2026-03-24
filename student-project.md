```python
# train_test_split: 데이터 나누기
# StandardScaler: 데이터 크기 맞추기
# LinearRegression: 예측 모델
# r2_score: 설명력 평가
# MSE: 큰 오차 중심 평가
# MAE: 평균 오차 평가
import pandas as pd
import numpy as np
```


```python
df=pd.read_csv(r"/kaggle/input/student-performance-predictions/student_performance.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StudentID</th>
      <th>Name</th>
      <th>Gender</th>
      <th>AttendanceRate</th>
      <th>StudyHoursPerWeek</th>
      <th>PreviousGrade</th>
      <th>ExtracurricularActivities</th>
      <th>ParentalSupport</th>
      <th>FinalGrade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>John</td>
      <td>Male</td>
      <td>85</td>
      <td>15</td>
      <td>78</td>
      <td>1</td>
      <td>High</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Sarah</td>
      <td>Female</td>
      <td>90</td>
      <td>20</td>
      <td>85</td>
      <td>2</td>
      <td>Medium</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Alex</td>
      <td>Male</td>
      <td>78</td>
      <td>10</td>
      <td>65</td>
      <td>0</td>
      <td>Low</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Michael</td>
      <td>Male</td>
      <td>92</td>
      <td>25</td>
      <td>90</td>
      <td>3</td>
      <td>High</td>
      <td>92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Emma</td>
      <td>Female</td>
      <td>88</td>
      <td>18</td>
      <td>82</td>
      <td>2</td>
      <td>Medium</td>
      <td>85</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Olivia</td>
      <td>Female</td>
      <td>95</td>
      <td>30</td>
      <td>88</td>
      <td>1</td>
      <td>High</td>
      <td>90</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Daniel</td>
      <td>Male</td>
      <td>70</td>
      <td>8</td>
      <td>60</td>
      <td>0</td>
      <td>Low</td>
      <td>62</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Sophia</td>
      <td>Female</td>
      <td>85</td>
      <td>17</td>
      <td>77</td>
      <td>1</td>
      <td>Medium</td>
      <td>78</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>James</td>
      <td>Male</td>
      <td>82</td>
      <td>12</td>
      <td>70</td>
      <td>2</td>
      <td>Low</td>
      <td>72</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Isabella</td>
      <td>Female</td>
      <td>91</td>
      <td>22</td>
      <td>86</td>
      <td>3</td>
      <td>High</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 9 columns):
     #   Column                     Non-Null Count  Dtype 
    ---  ------                     --------------  ----- 
     0   StudentID                  10 non-null     int64 
     1   Name                       10 non-null     object
     2   Gender                     10 non-null     object
     3   AttendanceRate             10 non-null     int64 
     4   StudyHoursPerWeek          10 non-null     int64 
     5   PreviousGrade              10 non-null     int64 
     6   ExtracurricularActivities  10 non-null     int64 
     7   ParentalSupport            10 non-null     object
     8   FinalGrade                 10 non-null     int64 
    dtypes: int64(6), object(3)
    memory usage: 848.0+ bytes


****
### Data Visualization
****


```python
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
# 성별, 방과후 활동, 부모님 지원의 분포를 보여주는 파이 차트
# 사용된 변수: 'Gender', 'ExtracurricularActivities', 'ParentalSupport'
tdf=df['Gender'].value_counts().reset_index()
tdf.columns=['Gender','count']
tdf1=df['ExtracurricularActivities'].value_counts().reset_index()
tdf1.columns=['ExtracurricularActivities','count']
tdf2=df['ParentalSupport'].value_counts().reset_index()
tdf2.columns=['ParentalSupport','count']
fig,axs=plt.subplots(1,3,figsize=(13,13))
axs[0].set_title("Distribution of Gender")
axs[0].pie(x=tdf['count'],labels=tdf['Gender'],autopct='%.2f%%')
axs[0].legend(tdf['Gender'])
axs[1].set_title("Distribution of Extra Curricular Activities")
axs[1].pie(x=tdf1['count'],labels=tdf1['ExtracurricularActivities'],autopct='%.2f%%')
axs[1].legend(tdf1['ExtracurricularActivities'])
axs[2].set_title("Distribution of Parental Support")
axs[2].pie(x=tdf2['count'],labels=tdf2['ParentalSupport'],autopct='%.2f%%')
axs[2].legend(tdf2['ParentalSupport'])
plt.tight_layout()
plt.show()
```


    
![png](student-performance%20%281%29_files/student-performance%20%281%29_5_0.png)
    


**결과 해석:**
1. 이 데이터셋에는 남학생과 여학생의 수가 동일하게 분포되어 있습니다.
2. 약 80%의 학생이 방과후 활동에 참여하고 있으며, 그중 20%는 최대치인 3가지 종류의 방과후 활동을 병행하고 있습니다.
3. 약 70%의 학생들이 중간 수준 이상의 높은 부모님 지원을 받고 있습니다. 반면 나머지 30%는 부모님의 지원이 다소 부족한 편입니다.

**전반적으로, 데이터는 학생들이 방과후 활동에 적극적으로 참여하고 있으며 많은 학생들이 부모님의 긍정적인 지원을 받고 있음을 보여줍니다.**


```python
# 성별로 구분하여 방과후 활동과 부모님 지원을 비교하는 카운트 플롯(막대 그래프)
# 사용된 변수: 'ExtracurricularActivities', 'ParentalSupport', 'Gender'
fig,axs=plt.subplots(1,2,figsize=(15,6))
sns.countplot(x=df['ExtracurricularActivities'],hue=df['Gender'],ax=axs[0])
sns.countplot(x=df['ParentalSupport'],hue=df['Gender'],ax=axs[1])
plt.tight_layout()
plt.show()
```


    
![png](student-performance%20%281%29_files/student-performance%20%281%29_7_0.png)
    


**결과 해석:**

**방과후 활동 vs 성별 (왼쪽 그래프):**
1. 여학생에 비해 남학생이 방과후 활동에 전혀 참여하지 않는(0개) 비율이 훨씬 높습니다.
2. 여학생은 보통 1~2개의 방과후 활동에 활발히 참여하고 있음을 보여줍니다.
3. 3개의 활동을 하는 비율은 남녀가 비슷합니다.
*결론:* 여학생은 전반적으로 방과후 활동에 잘 참여하지만, 남학생은 아무런 활동에도 참여하지 않는 경향이 더 큰 편입니다.

**부모님 지원 vs 성별 (오른쪽 그래프):**
1. 부모님의 적극적 지원(High)을 받는 비율은 남녀가 비슷하게 나타납니다.
2. 여학생은 중간 정도의 지원(Medium)을 압도적으로 많이 받고 있습니다.
3. 남학생만이 부모님의 지원이 부족한(Low) 범주에 속해 있습니다.

*결론:* 부모님의 지원 정도는 방과후 활동 참여도로 이어집니다. 부모님의 지원이 저조한 그룹에 속하는 남학생들이 방과후 활동에 덜 참여하게 된 원인 중 하나로 유추해 볼 수 있습니다.


```python
# 성별, 방과후 활동, 부모님 지원에 따른 최종 성적의 분포를 분석하는 박스 플롯
# 사용된 변수: 'Gender', 'ExtracurricularActivities', 'ParentalSupport', 'FinalGrade'
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(x='Gender', y='FinalGrade', data=df, ax=axs[0],hue='Gender')
axs[0].set_title('Final Grade by Gender')
axs[0].set_xlabel('Gender')
axs[0].set_ylabel('Final Grade')

sns.boxplot(x='ExtracurricularActivities', y='FinalGrade', data=df, ax=axs[1],hue='ExtracurricularActivities')
axs[1].set_title('Final Grade by Extracurricular Activities')
axs[1].set_xlabel('Extracurricular Activities')
axs[1].set_ylabel('Final Grade')

sns.boxplot(x='ParentalSupport', y='FinalGrade', data=df, ax=axs[2],hue='ParentalSupport')
axs[2].set_title('Final Grade by Parental Support')
axs[2].set_xlabel('Parental Support')
axs[2].set_ylabel('Final Grade')

plt.tight_layout()
plt.show()
```


    
![png](student-performance%20%281%29_files/student-performance%20%281%29_9_0.png)
    


**결과 해석:**

**성별에 따른 최종 성적:** 여학생이 남학생보다 전반적으로 높은 성과(중간값)를 보이며, 하위권으로 떨어지는 성적 편차도 더 적습니다.
**방과후 활동에 따른 최종 성적:** 방과후 활동 참여 개수가 많을수록(특히 3개) 최종 성적이 확연히 높아지는 뚜렷한 경향을 보입니다. 어떠한 활동도 하지 않는 학생들의 최종 성적이 가장 낮습니다.
**부모님 지원에 따른 최종 성적:** 부모님의 지원이 높을수록 최종 성적이 더 높습니다. 특히 지원이 높은 그룹일수록 성적의 편차가 작아(학업성취에 일관성을 보임) 학업에 매우 긍정적입니다.

**종합:** 성별, 방과후 활동, 부모님 지원 모두 학생의 최종 성적에 매우 유의미한 영향을 미칩니다. 부모님의 든든한 지원과 활발한 활동이 좋은 성적으로 뚜렷하게 이어짐을 알 수 있습니다.


```python
# 주당 공부 시간과 성적(이전 성적 및 최종 성적) 간의 관계를 보여주는 산점도
# 사용된 변수: 'StudyHoursPerWeek', 'PreviousGrade', 'FinalGrade'
plt.figure(figsize=(15, 4))
sns.scatterplot(data=df, x='StudyHoursPerWeek', y='PreviousGrade', label='Previous Grade')
sns.scatterplot(data=df, x='StudyHoursPerWeek', y='FinalGrade', label='Final Grade')
plt.title("Study Hours Per week vs Final and Previous Grades")
plt.legend()
plt.show()
```


    
![png](student-performance%20%281%29_files/student-performance%20%281%29_11_0.png)
    


**결과 해석:**

1. 주당 공부 시간이 많을수록 이전 성적(Previous Grade)과 최종 성적(Final Grade) 모두 뚜렷하게 높아지는 강한 양의 상관관계를 보입니다.
2. 이전 성적이 높았던 학생들이 꾸준히 충분한 공부 시간을 확보하며 최종 성적에서도 우수한 결과를 지속함(성실함)을 그래프를 통해 유추할 수 있습니다.


```python
# 공부 시간, 출석률, 방과후 활동이 최종 성적에 미치는 영향을 성별로 색상 구분하여 보여주는 산점도
# 사용된 변수: 'StudyHoursPerWeek', 'AttendanceRate', 'ExtracurricularActivities', 'FinalGrade', 'Gender'
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

sns.scatterplot(x=df['StudyHoursPerWeek'], y=df['FinalGrade'], ax=axs[0],hue=df['Gender'])
axs[0].set_title('Study Hours vs Final Grade')
axs[0].set_xlabel('Study Hours Per Week')
axs[0].set_ylabel('Final Grade')


sns.scatterplot(x=df['AttendanceRate'], y=df['FinalGrade'], ax=axs[1],hue=df['Gender'])
axs[1].set_title('Attendance Rate vs Final Grade')
axs[1].set_xlabel('Attendance Rate')
axs[1].set_ylabel('Final Grade')


sns.scatterplot(x=df['ExtracurricularActivities'], y=df['FinalGrade'], ax=axs[2],hue=df['Gender'])
axs[2].set_title('Extracurricular Activities vs Final Grade')
axs[2].set_xlabel('Extracurricular Activities')
axs[2].set_ylabel('Final Grade')

plt.tight_layout()
plt.show()
```


    
![png](student-performance%20%281%29_files/student-performance%20%281%29_13_0.png)
    


**결과 해석:**

**공부 시간 vs 최종 성적:** 공부 시간이 늘어날수록 최종 성적이 상승합니다. 여학생들이 남학생보다 대체적으로 더 많은 시간을 공부에 할애하고 결과 역시 좋습니다.
**출석률 vs 최종 성적:** 출석률이 높을수록 최종 성적이 우수한 강한 비례 관계를 보입니다. 앞선 통계대로 여학생들이 남학생보다 출석률이 높고 성적이 좋은 편입니다.
**방과후 활동 vs 최종 성적:** 다른 두 지표(공부 시간, 출석률) 만큼 절대적인 선형 기울기를 띠진 않지만, 역시 방과후 활동이 잦을수록 성적이 전반적으로 향상되는 경향을 확인할 수 있습니다.


```python
# 모든 수치형 변수에 대한 상관관계 히트맵
# 사용된 변수: 모든 수치형 열 (int64)
plt.figure(figsize=(12,4))
tdf=df.select_dtypes(include=['int64'])
sns.heatmap(tdf.corr(),annot=True)
plt.show()
```


    
![png](student-performance%20%281%29_files/student-performance%20%281%29_15_0.png)
    


****
### Data Preprocessing
****


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 9 columns):
     #   Column                     Non-Null Count  Dtype 
    ---  ------                     --------------  ----- 
     0   StudentID                  10 non-null     int64 
     1   Name                       10 non-null     object
     2   Gender                     10 non-null     object
     3   AttendanceRate             10 non-null     int64 
     4   StudyHoursPerWeek          10 non-null     int64 
     5   PreviousGrade              10 non-null     int64 
     6   ExtracurricularActivities  10 non-null     int64 
     7   ParentalSupport            10 non-null     object
     8   FinalGrade                 10 non-null     int64 
    dtypes: int64(6), object(3)
    memory usage: 848.0+ bytes



```python
df.drop(columns=['StudentID','Name'],inplace=True,axis=1)
print(df['Gender'].value_counts())
```

    Gender
    Male      5
    Female    5
    Name: count, dtype: int64



```python
df['Gender']=df['Gender'].apply(lambda x: 1 if x=='Male' else 0)
print(df['ParentalSupport'].value_counts())
```

    ParentalSupport
    High      4
    Medium    3
    Low       3
    Name: count, dtype: int64



```python
df['ParentalSupport']=df['ParentalSupport'].apply(lambda x: 2 if x=='High' else 1 if x=='Medium' else 0)
print(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 7 columns):
     #   Column                     Non-Null Count  Dtype
    ---  ------                     --------------  -----
     0   Gender                     10 non-null     int64
     1   AttendanceRate             10 non-null     int64
     2   StudyHoursPerWeek          10 non-null     int64
     3   PreviousGrade              10 non-null     int64
     4   ExtracurricularActivities  10 non-null     int64
     5   ParentalSupport            10 non-null     int64
     6   FinalGrade                 10 non-null     int64
    dtypes: int64(7)
    memory usage: 688.0 bytes
    None


**결과 해석 (데이터 전처리 2):**

부모님의 지원 정도를 나타내는 `ParentalSupport` 범주형 변수를 수치형으로 변환했습니다. 
머신러닝 모델이 값의 크고 작음을 이해할 수 있도록 지원이 높음('High')은 2로, 중간('Medium')은 1로, 낮음('Low')은 0으로 매핑하여 순서가 있는 수치형 데이터(Ordinal variables)로 전처리를 완료했습니다.


```python
# 전처리 후 모든 변수에 대한 상관관계 히트맵
# 사용된 변수: 모든 열
plt.figure(figsize=(12,4))
sns.heatmap(df.corr(),annot=True)
plt.show()
```


    
![png](student-performance%20%281%29_files/student-performance%20%281%29_22_0.png)
    


****
### Models
****


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
```

#### Data Preparation and Standardization


```python
x=df.drop(columns='FinalGrade')
y=df['FinalGrade']
x_t,x_te,y_t,y_te=train_test_split(x,y,test_size=0.25,random_state=20)
ss=StandardScaler()
x_t=ss.fit_transform(x_t)
x_te=ss.transform(x_te)
```

**결과 해석 (추가된 전처리 과정):**

1. **데이터 스플릿 (Train-Test Split):** 전체 데이터를 모델 학습용(Training set) 75%와 검증용(Testing set) 25% 비율로 나누었습니다. 이를 통해 편향되지 않고 일반화된 모델을 평가할 수 있습니다.
2. **데이터 스케일링 (Standardization):** 각 특성(feature)들의 범위를 동일하게 맞추는 표준화 작업을 진행했습니다. 이를 통해 값의 크기가 지나치게 크거나 작은 특정 변수가 결과에 과도한 영향을 미치지 않도록 방지합니다.

#### Linear Regression


```python
reg=LinearRegression()
reg.fit(x_t,y_t)
pred1=reg.predict(x_te)
pred2=reg.predict(x_t)
```


```python
print("Training Metrics")
print('R2 Score: ',r2_score(y_t,pred2))
print('Mean Squared Error: ',mean_squared_error(y_t,pred2))
print('Mean Absolute Error: ',mean_absolute_error(y_t,pred2))
```

    Training Metrics
    R2 Score:  1.0
    Mean Squared Error:  8.654931074424815e-29
    Mean Absolute Error:  6.090366306515144e-15



```python
print("Testing Metrics")
print('R2 Score: ',r2_score(y_te,pred1))
print('Mean Squared Error: ',mean_squared_error(y_te,pred1))
print('Mean Absolute Error: ',mean_absolute_error(y_te,pred1))
```

    Testing Metrics
    R2 Score:  0.9522714724399345
    Mean Squared Error:  1.8136840472824893
    Mean Absolute Error:  1.3287671232876666


**결과 해석 (Metrics Explanation):**

*   **R2 Score (결정계수): 0.952**
    *   R2 점수는 모델이 데이터의 분산을 얼마나 잘 설명하는지를 0에서 1 사이의 비율로 나타냅니다. 0.95라는 수치는 모델이 종속 변수(최종 성적) 변동성의 약 95%를 설명할 수 있음을 의미하며, 이는 예측 성능이 매우 뛰어남을 보여줍니다.
*   **Mean Squared Error (MSE, 평균 제곱 오차): 1.813 & Mean Absolute Error (MAE, 평균 절대 오차): 1.328**
    *   MSE와 MAE는 1이 넘는 수치가 나왔는데, 이는 R2와 다른 척도를 사용하기 때문입니다. R2가 단순한 '비율(0~1)'인 반면, MSE와 MAE는 **실제 데이터(성적)와 같은 단위의 오차 값**을 나타냅니다. 
    *   즉, MAE가 1.32라는 것은 우리 모델이 성적을 예측할 때 평균적으로 약 1.32점 정도의 오차가 발생한다는 뜻입니다. 종속 변수인 Final Grade의 범위(대체로 0~100 또는 0~20 등의 점수 척도)를 고려하면 1.32점의 오차는 꽤 작고 훌륭한 수준이라고 볼 수 있습니다.


```python
# 모델의 잔차에 대한 분포도 (KDE 플롯)
# 사용된 변수: 'y_te' (실제 FinalGrade), 'pred1' (예측된 FinalGrade)
residuals=y_te-pred1
sns.displot(residuals, kind='kde', height=5, aspect=3)
plt.title("Residuals")
plt.show()
```

    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):



    
![png](student-performance%20%281%29_files/student-performance%20%281%29_33_1.png)
    


**결과 해석 (마지막 잔차 플롯):**

잔차(실제 성적에서 예측 성적을 뺀 값)의 분포를 나타내는 그래프입니다. 예측 모델이 얼마나 정밀한지를 보여주는 지표로, 그래프가 0점을 중심으로 좁은 정규분포(종 모양) 형태를 띨수록 오차가 적어 모델 성능이 훌륭함을 의미합니다. 이 플롯은 잔차가 0 구간에 매우 밀집해 있어 우리 모델이 실제 점수를 상당히 정확하게 예측하고 있음을 재확인할 수 있습니다.

**Provide Suggestions and Kindly Upvote :)**
