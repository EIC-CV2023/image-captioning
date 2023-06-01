# robocup2023-cv-image-captioning
Prior to using image captioning, BLIP need to be cloned and installed
<pre>
git clone https://github.com/salesforce/BLIP.git
cd BLIP
pip install -r requirements.txt
cd ..
</pre>

## Inference
run this first
<pre>
python main.py
</pre>
Then run following command in new terminal
<pre>
python client.py
</pre>

get age and gender caffemodel from https://drive.google.com/file/d/1RcEI4lk6FesPCBwHi5xAY_svZ8It7XtF/view?usp=sharing
download and extract age_caffe.caffemodel and gender_caffe.caffemodel to age_gender_recog folder
