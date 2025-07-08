# Start with neurodebian xenial, using python3

FROM python:3.10-slim

# Install OS-level dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget curl bc libglu1 libxext6 libsm6 libxrender1 libfontconfig1 libxcursor1 \
    libxft2 libxinerama1 libxrandr2 libxi6 libgomp1 libexpat1 libpng16-16 \
    tar bzip2 && \
    apt-get clean

# Set FSL environment variables
ENV FSLDIR=/opt/fsl
ENV PATH="${FSLDIR}/bin:$PATH"
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Install FSL 6.0.6 (non-interactive, headless)
RUN curl -sSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py -o fslinstaller.py && \
    python3 fslinstaller.py -d /opt/fsl -V 6.0.6 -q --nocheck && \
    rm fslinstaller.py

# Install packages
RUN pip3 install --upgrade pip==25.1.1
RUN pip3 install numpy==2.2.6
RUN pip3 install nibabel==5.3.2
RUN pip3 install nipype==1.10.0

# Make directory for flywheel spec
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
COPY run.py ${FLYWHEEL}/run
COPY manifest.json ${FLYWHEEL}/manifest.json

# Put script into flywheel folder
WORKDIR ${FLYWHEEL}
COPY t1fit_unwarp.py t1_fitter.py ./
COPY fsl-fmap-correction.sh ./fsl-fmap-correction
RUN chmod +x run t1fit_unwarp.py t1_fitter.py fsl-fmap-correction

# Set the entrypoint
ENTRYPOINT ["/flywheel/v0/run"]
