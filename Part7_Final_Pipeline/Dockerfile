# Use the basic Python 3 image as launching point
FROM python:3.6.3

# Add the script or text to the Dockerfile
ADD Final_Pipeline.py /home
ADD energydata_complete.csv /home
# ADD requirements.txt /home

# Install required Libraries
# RUN pip install -r ./home/requirements.txt
RUN pip install numpy
RUN pip install pandas
RUN pip install seaborn
RUN pip install sklearn
RUN pip install scipy

CMD ["python3", "/home/Final_Pipeline.py"] 
