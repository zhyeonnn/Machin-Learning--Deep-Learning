# pandas란?

|Series|DataFrame|
|:----------:|:------:|
|1차원|2차원|

# 데이터프레임 DataFrame
- **데이터 불러오기** -> `pd.read_csv("파일명")`
- **데이터 내보내기** -> `pd.to_csv("파일명")` <br>
데이터 내보낼 때 `pd.to_csv("파일명", index = False)`index = False를 안 넣으면 고유 행번호가 맨 앞 열에 추가된다. 주의하기 !!<br>
  <img width="424" height="259" alt="image" src="https://github.com/user-attachments/assets/6b4a5c7e-9e14-40f2-b5b0-14d7b939c257" />

- **데이터 특성** -> `.dtypes`
- **열이름 조회** -> `.columns`
- **.index**
  -> `.index`<br>
    .index(2)는 3번째 행을 추출할 수 있다. 위에서 index = False처럼 index가 나타내는 고유번호 정보를 나타낸다. 
- **기초 통계 정보** -> `.describe()`
  평균, 최대, 최소, 표준편차의 값을 알려준다.<br>
- **데이터프레임의 정보** -> `.info()`<br>
  <img width="301" height="205" alt="image" src="https://github.com/user-attachments/assets/a17c4bfa-8365-44e2-85e7-b57acda7b0e3" />

  index는 몇번까지 있고, null이 아닌 값이 몇개 존재하며 객체인지 숫자 형태인지 보여준다.<br>
- **데이터 첫부분** -> `data.head()`   or   `data.head(7)`<br>
  기본값이 5행까지 출력이고 head(7)이면 7행까지 출력된다.<br>
- **데이터 끝부분** -> `data.tail()`   or   `data.tail(3)`
  기본값이 5행까지 출력이고 tail(7)이면 7행까지 출력된다.<br>
- **.iloc**<br>
  -> `data.iloc[:3]`<br>
    인덱스 번호 상관없이 열 순서가 0부터 3까지의 행, 3미포함<br>
  -> `data.iloc[3]`<br>
    열 순서가 3인 행<br>
- **.loc**<br>
  -> `data.loc[3]`<br>
    인덱스가 3인 행, 3인 행이 2개 이상일 수도 있음<br>
  -> `data.loc['A201']`<br>
    인덱스가 A201인 행, 문자도 가능<br>
- **열이름의 행 조회**<br>
  -> `data.Make` : Make라는 열이름의 행 출력<br>
  -> `data['Make']` : Make라는 열이름의 행 출력, 같지만 방식이 다르다. <br>
  BUT<br>
  -> `data['price (10)]'` : price (10)이라는 열이름의 형 출력<br>
  -> `data.price (10)]` :  ERROR, price (10)이라는 열이름은 띄워쓰기가 존재하기 때문에 이 방식으로는 사용할 수 없다.<br>
- **연산**<br>
  -> `car_sales[car_sales['Make'] == 'Toyota']`<br>
    `data[연산 조건]`으로 작성하면 데이터프레임 형식으로 출력된다.<br>

- **크로스탭**<br>
  -> `pd.crosstab(car_sales['Make'],car_sales['Doors'])`<br>
    행(Make)과 열(Doors)을 통해 테이블을 만든다, 값은 행과 열에 맞는 개수<br>
- **plot()**<br>
  -> `car_sales['Odometer (KM)'].plot()` : Odometer (KM)의 데이터를 그래프로 나타낸다.<br>
- **hist()**<br>
  -> `car_sales['Odometer (KM)'].hist()` : Odometer (KM) 값의 빈도를 나타낸다.<br>
- **데이터타입 변환(object -> int)**
  ```python
    car_sales['Price'] = (
    car_sales['Price']
    .str.replace(r'[\$,]', '', regex=True)   # $와 ,만 제거
    .astype(float)                           # float 변환 (소수 포함)
    .astype(int)                             # 필요하면 int 변환
    )
  ```
  `regex = True`는 정규표현식을 사용한다는 의미이고 regex를 사용할 때 `r''`을 통해 여러 문자나 패턴을 한번에 처리할 수 있다.<br>
- **소문자, 대문자 변환**<br>
  -> `data.str.lower()` / `data.str.upper()`<br>
- **fillna()**<br>
  -> **결측치 변경하기**
  -> `car_sales_missing["Odometer"].fillna(car_sales_missing['Odometer'].mean())` : Odometer의 결측치를 찾아서 Odometer의 평균값으로 채워라<br>
  -> `.fillna(car_sales_missing['Odometer'].mean(),inplace= True)` : `inplace = True`를 설정하면 재배열없이 사용가능하다.<br>
  -> `car_sales_missing["Odometer"] = car_sales_missing["Odometer"].fillna(car_sales_missing['Odometer'].mean()) ` <br>: `inplace = True`를 사용하지 않으면 이렇게 재배열하기<br>
- **dropna()**<br>
  -> **결측치 제거하기**<br>
  ->  `car_sales_missing.dropna(inplace=True)` : fillna와 마찬가지로 `nplace = True`를 사용하면 재배열 안 해도 된다.<br>


# Series
- `pd.Series([5,5,5,5,5,5])`<br>
  -> 1차원으로 배열<br>
- **열 생성**<br>
  -> `car_sales['Number of wheels'] = 4` : 이렇게 사용하면 1차원 배열의 모든 행에 4가 들어가게된다.<br>
- **drop()**<br>
  -> `car_sales.drop('Total fuel used',axis = 1)` : axis = 1은 열, axis = 0은 행을 지울 것이라는 의미이다.
- **sample()**
  -> ` car_sales.sample(frac =1)`에서 데이터를 섞어서 frac = 0 ~ 1사이의 값으로 데이터를 얼마나 사용할 것인지를 정하는 것이다.<br>
    frac = 0.2로 설정하여 전체 데이터들의 20%만 랜덤으로 추출하여 사용하게 된다.<br>
    이것을 샘플데이터라고도 한다.<br>
- **reset_index()**
  -> sample()을 사용해서 랜덤으로 섞은 데이터를 다시 초기화(인덱스 순서대로)할 수 있다.<br>
  -> 기존에 있던 인덱스와 새로 생긴 인덱스 두개의 열이 생기는데 기존의 인덱스를 지우는 방법은 reset_index에 `drop=True, inplace= True`를 추가하면 된다.<br>
  
  <img width="80%" height="262" alt="image" src="https://github.com/user-attachments/assets/59d0ef87-6f96-4b31-82aa-853a481a9d0a" />

  <img width="80%" height="259" alt="image" src="https://github.com/user-attachments/assets/83ed1689-85a1-4b01-8b80-7a1cd6d18ad6" />

- **apply()**
  -> apply의 괄호 안 내용의 결과를 반환하는 것이다.<br>
  -> ` car_sales['Odometer (KM)'].apply(lambda x:x /1.6)` : 익명의 값으로 계산을 해서 결과만 반환한다.
