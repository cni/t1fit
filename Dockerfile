# Start with neurodebian xenial, using python3

FROM neurodebian:xenial
MAINTAINER Hua Wu <huawu@stanford.edu>

# Install dependencies
RUN echo deb http://neurodeb.pirsquared.org data main contrib non-free >> /etc/apt/sources.list.d/neurodebian.sources.list
RUN echo deb http://neurodeb.pirsquared.org trusty main contrib non-free >> /etc/apt/sources.list.d/neurodebian.sources.list
RUN apt-get update && apt-get -y install \
    python3-dev python3-pip \
    fsl-5.0-core
RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.15.4 \
    && pip3 install nibabel==2.3.2
Run pip3 install nipype==1.1.7

# Make directory for flywheel spec
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
COPY run.py ${FLYWHEEL}/run
COPY manifest.json ${FLYWHEEL}/manifest.json

# Put script into flywheel folder
WORKDIR ${FLYWHEEL}
COPY t1fit_unwarp.py t1_fitter.py ./
RUN chmod +x run t1fit_unwarp.py t1_fitter.py 

# Set the entrypoint
ENTRYPOINT ["/flywheel/v0/run"]
