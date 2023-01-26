FROM python:3.10.9-slim

WORKDIR /home

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git libgomp1 apt-utils libgl1

COPY id_rsa.pub /home/.ssh/id_rsa.pub

RUN echo "git clone git@github.com:onurcatmabacak/open3d_trimesh_scikit-fem_vs_pymesh.git" > /home/get_pymesh_test.sh
RUN echo "git clone https://github.com/nipy/mindboggle.git" >> /home/get_pymesh_test.sh
RUN echo "mv mindboggle/mindboggle /usr/local/lib/python3.10/site-packages/mindboggle" >> /home/get_pymesh_test.sh
RUN pip install open3d trimesh scikit-fem numpy scipy networkx 