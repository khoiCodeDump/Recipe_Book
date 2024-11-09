const CACHE_NAME = 'recipe-cache-v1';
const urlsToCache = [
  '/',
  '/static/home.css',
  '/static/recipe.css',
  '/static/header.css',
  '/static/images/food_image_empty.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
