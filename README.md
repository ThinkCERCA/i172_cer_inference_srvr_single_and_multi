# claim-and-multi-label-inference-testing
Download the **multi-label model** from google drive: https://drive.google.com/drive/u/0/folders/1alMb6xmH-Cx3VJRKfYW4UJ18bHeFcmQS  or 

download the **claim Xlnet model** form google drive: https://drive.google.com/drive/u/0/folders/1Iy-TqgGbU6odNKgj4EIIqdPpW3Zk-Xb-

$Window$:

Put model and py file in the same directory. Make sure each model has the right py file. Using cmd find the file directory and run python claim-xlnet-predict.py.


$Testing$:

Start a new terminal or cmd, run
***curl -X POST http://localhost:8008/predict -H "Content-Type:application/json" -d "{\"content\":\"the python\"}"***

Put anything you want to test in content instead the python


$linux$:

Put model and py file in the same directory. Make sure each model has the right py file. Run nohup python claim-xlnet-predict.py.

$Testing$:

Run ***curl -X POST http://localhost:8008/predict -H "Content-Type:application/json" -d "{"content":"the python"}"***

