<!-- readme.md -->

A Dashboard for momentum trading of the ASX

To deploy to AWS Elasticbeanstalk:

# create a new environment
conda create --name eb_py37 python=3.7
conda activate eb_py37

git init
git add .
git commit -m "some commit message"

pip install awsebcli
eb init

<!-- add the elasticbeanstalk config to git -->
git add .
git commit -m "another commit message"

<!-- then create the environment -->
eb create 

# to update deployment
git add .
git commit -m "update commit message:

eb deploy