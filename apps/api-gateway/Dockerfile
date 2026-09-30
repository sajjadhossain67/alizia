FROM node:22-alpine

WORKDIR /app

COPY package*.json ./
COPY packages ./packages
COPY apps/api-gateway ./apps/api-gateway

RUN npm install

EXPOSE 3000

CMD ["npm", "run", "dev:gateway"]
