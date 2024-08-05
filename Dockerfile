FROM python:3.10

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt 

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Set environment variables
ENV HOME=/usr/src/app

# Expose port (if needed)
EXPOSE 8000