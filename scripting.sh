#python main_train.py
#python main_test.py

#cd AutoSeg/Dicom
#python Mask2Contour.py

#python structure_comparison


cd AutoSeg/Dicom
python2 niimage.py
cd ../..
python2 main_train_gpu1.py

python2 SegThor_submit.py