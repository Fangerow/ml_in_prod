# Homework 1

## Shamankov Nikolay, MADE DS-12

### Getting started
```
pip3 install -r requirements.txt
```
### Model education with default parameters
```
python3 -m ml_project.scr.model_training_pipeline +ppl=train_pipeline_default.yaml
```
#### And the other way:
```
python3 -m ml_project.scr.model_training_pipeline +ppl=train_pipeline_special.yaml
```

#### Make predictions for data 
Usage:
```
python3 -m ml_project.scr.make_prediction <model_trainig artefacts <path to test data> <path to save predictions>
```

```
python3 -m ml_project.scr.make_prediction model.joblib ml_project/test.csv ml_project/result.csv
```


### Project stucture
Will be here later, with Cl probably.

### ������������� � ����������� �������:

-������ ������� �� ���� �������� �������� � ���������� ���������������. ��������� ��������: scr (��������������� ������� �������� ������ � ��������� ������������, ���������),
tests(����� ������� �  �������� ��������). ��������������� ���������� ��������� ��������� ����� ������, Eda � ��������. 
- ��� ���������������� �������������� `hydra`.
- ������� ������������ ��� �������� ������������� ��������� � k �������.
- ��� ������������ ������ �������� ������������ ������������� ������: ��� ����� ������ �������-������� �������� ������ ���������������. ����� �� ��������� ���������� ���������� ������������ ������������ ������ �� ������ ������� ��������.
- � ������� �������� "������" ���������������� ������ ������.
� ���������� � ���� ��������� ������������� ���������� ����� � html �����
- ��� ���������� ��������� �� �������� ���� �������� ����������.

### ����������

0. � �������� � ���� �������� ������� �������� "�������������" � ����������� �������, ������� ������� � ����� ������. (1/1)
1. � ����-�������� ��������� ���������� (1/1)
2. ��������� EDA � �������������� �������. (2	/2)
3. �������� �������/����� ��� ���������� ������, ����� �������� ��� ������� ��������� ������, �������� � readme ���������� �� ������� (3/3)
4. �������� �������/����� predict (����� �������� ��� ������� ��������� ������), ������� ������ �� ���� ��������/� �� ��������, �������� ������� (��� �����) � ������� ������� �� ��������� ����, ���������� �� ������ �������� � readme (3/3)
5. ������ ����� ��������� ���������. (2/2)
6. ������������ ������� (2/2)
7. �������� ����� �� ��������� ������ � �� ������ �������� � predict. (3/3)
8. ��� ������ ������������ ������������� ������, ������������ � ��������. (2/2)
9. �������� ������ ��������������� � ������� �������� yaml. (3/3) 
10. ������������ ���������� ��� ��������� �� �������, � �� ����� dict (1/2)
(�������� ������������� ���� ������ ����� ������ ��� ������� ��� �������� ������)
11. �������� ��������� ����������� � ������������� ��� (3/3)
12. � ������� ������������� ��� ����������� (1/1)
13. �������� CI ��� ������� ������, ������� �� ������ github actions (0/3).

�������������� �����:
- ����������� hydra ��� ���������������� (3/3)
- Mlflow (0/6)