# ml_rev
# ml_basics

## dagsshub process 

set MLFLOW_TRACKING_URI =https://dagshub.com/harsh9769/ml_basics.mlflow

set MLFLOW_TRACKING_USERNAME=harsh9769

set MLFLOW_TRACKING_PASSWORD= 1577816e26554c74a3a2cc6ccd9d0890c5c17ca7

## AWS setup

1.Login to AWS console
2.CReate IAM user
3.Export credential in your AWS cli by running "aws configure"
4.creare a s3 bucket
5.create a ec2 machine ubuntu & add security groups 5000 port

## Run following code on EC2 machine

sudo apt update

sudo apt install python3-pip

sudo pip3 install pipenv

sudo pip3 install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell

aws configure

