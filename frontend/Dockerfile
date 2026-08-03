# microgpt frontend — statikus nginx
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html viz.html style.css app.js viz.js /usr/share/nginx/html/
COPY fonts /usr/share/nginx/html/fonts/
