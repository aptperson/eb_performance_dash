<!-- readme.md -->

[# A Dashboard for momentum trading of the ASX](http://performance-dash-dev.ap-southeast-2.elasticbeanstalk.com/)

<!-- [APTCapital Performance dash] -->

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
follow the prompts

#### much success

#### to update deployment
git add .
git commit -m "update commit message:

eb deploy

#### to remove the application and environment
eb temimnate

[inspired by](https://medium.com/@austinlasseter/plotly-dash-and-the-elastic-beanstalk-command-line-89fb6b67bb79)


#### Disclaimer
This website and the information contained within is for general information purposes only. It is not a source of legal, financial or investment advice. For legal, financial or investment advice consult a qualified and registered practitioner
