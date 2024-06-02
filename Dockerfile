FROM node:20.14

WORKDIR /usr/src/app

COPY package.json package-lock.json ./

RUN npm install

COPY . .
