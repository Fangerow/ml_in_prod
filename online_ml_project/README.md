# Homework 2

## Shamankov Nikolay, MADE DS-12

### to Build docker image
```
cd online_ml_project/
docker build -t mlops_hw2:v8 .
```

Or with docker hub:
```
docker login --username dmosk
docker pull fangerow/fangerow:mlops_hw2
```

### How to run app with docker
```
docker run -d mlops_hw2:v8
docker ps
docker exec -it container_id bash
```
#### Request execute inside container:
```
python3 -m scr.requests_api
```

#### Run tests
```
python -m pytest test_app.py
```

### Оптимизация для образа Docker:

-Использован минимум зависимостей
- Использован легковесный (базовый для версии 3.9) образ python3.9-slim
