FROM pytorch/pytorch:latest

# make ssh key trick to clone private repository
ENV TZ=Asia/Tokyo
RUN apt-get -y update && apt-get -y install ssh && apt-get -y install git
RUN apt-get -y install emacs
RUN pip install astropy
RUN pip install matplotlib
RUN mkdir ~/.ssh
ARG ssh_prv_key
ARG ssh_pub_key
RUN git clone git@github.com:mpaillassa/HSC_unrecognized_blends.git
