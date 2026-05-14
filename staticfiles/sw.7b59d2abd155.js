// Vamos Frotas SLA - Service Worker
// Progressive Web App - Offline Support

const CACHE_NAME = 'vamos-frotas-sla-v1';
const STATIC_CACHE = 'vamos-static-v1';
const DYNAMIC_CACHE = 'vamos-dynamic-v1';

// Assets to cache on install
const STATIC_ASSETS = [
    '/',
    '/static/css/premium-dark.css',
    '/static/css/pwa-mobile.css',
    '/static/js/pwa-mobile.js',
    '/static/manifest.json'
];

// CDN assets to cache (with no-cors mode)
const CDN_ASSETS = [
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('[Service Worker] Installing...');
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => {
                console.log('[Service Worker] Caching static assets');
                
                // Cache same-origin assets normally
                const sameOriginPromise = cache.addAll(STATIC_ASSETS)
                    .catch(err => console.warn('[Service Worker] Some same-origin assets failed:', err));
                
                // Cache CDN assets with no-cors mode
                const cdnPromise = Promise.all(
                    CDN_ASSETS.map(url => 
                        cache.add(new Request(url, { mode: 'no-cors' }))
                            .catch(err => console.warn('[Service Worker] Failed to cache:', url))
                    )
                );
                
                return Promise.all([sameOriginPromise, cdnPromise]);
            })
            .then(() => {
                console.log('[Service Worker] Installed successfully');
                return self.skipWaiting(); // Activate immediately
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[Service Worker] Activating...');
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => {
                            return name !== STATIC_CACHE && name !== DYNAMIC_CACHE;
                        })
                        .map((name) => {
                            console.log('[Service Worker] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => {
                console.log('[Service Worker] Activated');
                return self.clients.claim(); // Take control immediately
            })
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Allowed CDN domains
    const allowedCDNs = [
        'cdn.jsdelivr.net',
        'fonts.googleapis.com',
        'fonts.gstatic.com'
    ];

    // Skip cross-origin requests except for allowed CDN assets
    const isAllowedCDN = allowedCDNs.some(cdn => url.hostname.includes(cdn));
    if (url.origin !== location.origin && !isAllowedCDN) {
        return;
    }

    // Network first strategy for API calls and dynamic content
    if (request.url.includes('/api/') || request.method !== 'GET') {
        event.respondWith(
            fetch(request)
                .then((response) => {
                    // Clone the response
                    const responseClone = response.clone();
                    // Cache successful responses
                    if (response.status === 200) {
                        caches.open(DYNAMIC_CACHE).then((cache) => {
                            cache.put(request, responseClone);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Return cached version if available
                    return caches.match(request);
                })
        );
        return;
    }

    // Cache first strategy for static assets
    event.respondWith(
        caches.match(request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    // Return cached version and update cache in background
                    fetch(request).then((response) => {
                        if (response.status === 200) {
                            caches.open(DYNAMIC_CACHE).then((cache) => {
                                cache.put(request, response);
                            });
                        }
                    }).catch(() => {
                        // Ignore network errors when updating cache
                    });
                    return cachedResponse;
                }

                // Not in cache, fetch from network
                return fetch(request)
                    .then((response) => {
                        // Check if valid response
                        if (!response || response.status !== 200) {
                            return response;
                        }

                        // Clone the response
                        const responseClone = response.clone();

                        // Add to cache
                        caches.open(DYNAMIC_CACHE).then((cache) => {
                            cache.put(request, responseClone);
                        });

                        return response;
                    })
                    .catch(() => {
                        // Network error, show offline page if HTML
                        if (request.headers.get('accept').includes('text/html')) {
                            return new Response(
                                '<html><body style="background: #0f1419; color: white; font-family: sans-serif; text-align: center; padding-top: 100px;"><h1>Você está offline</h1><p>Conecte-se à internet para acessar o Vamos Frotas SLA</p></body></html>',
                                { headers: { 'Content-Type': 'text/html' } }
                            );
                        }
                    });
            })
    );
});

// Handle messages from client
self.addEventListener('message', (event) => {
    if (event.data === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});

// Background sync (for future implementation)
self.addEventListener('sync', (event) => {
    console.log('[Service Worker] Background sync:', event.tag);
    // Could be used to sync data when connection is restored
});

// Push notification (for future implementation)
self.addEventListener('push', (event) => {
    console.log('[Service Worker] Push notification received');
    // Could be used for real-time notifications
});
