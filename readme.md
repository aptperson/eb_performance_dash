<!-- readme.md -->

# A Dashboard for momentum trading of the ASX

[APTCapital Performance dash](http://performance-dash-dev.ap-southeast-2.elasticbeanstalk.com/)

### To deploy to AWS Elasticbeanstalk:

#### create a new environment
conda create --name eb_py37 python=3.7
conda activate eb_py37

#### initialise git
git init
git add .
git commit -m "some commit message"

#### install AWS elabticbeanstalk cli and initilize
pip install awsebcli
eb init

#### add the elasticbeanstalk config to git
git add .
git commit -m "another commit message"

#### then create the environment
eb create 

#### much success

#### to update deployment
git add .
git commit -m "update commit message:

eb deploy