from django.test import TestCase, override_settings
from django.urls import reverse
from django.contrib.auth.models import User


class SinistroManualViewTestCase(TestCase):
    """Tests for the SinistroManualView"""
    
    def setUp(self):
        """Create a test user"""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.url = reverse('procedures-manual')
    
    def test_view_requires_authentication(self):
        """Test that unauthenticated users are redirected"""
        response = self.client.get(self.url)
        # Should redirect (either 301 for trailing slash or 302 for login)
        self.assertIn(response.status_code, [301, 302])
        # Check that redirect URL contains either the same path (for trailing slash)
        # or login (for authentication)
        self.assertTrue(
            '/api/procedures/manual' in response.url or '/login/' in response.url,
            f"Unexpected redirect URL: {response.url}"
        )
    
    def test_authenticated_user_can_access(self):
        """Test that authenticated users can access the view"""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(self.url, follow=True)
        # Should eventually get a 200 after following redirects
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'sinistro.html')
    
    def test_view_uses_correct_template(self):
        """Test that the view uses the sinistro.html template"""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(self.url, follow=True)
        self.assertTemplateUsed(response, 'sinistro.html')
    
    def test_view_url_resolves_correctly(self):
        """Test that the URL resolves to the correct path"""
        # The URL should be at /api/procedures/manual/
        expected_path = '/api/procedures/manual/'
        self.assertEqual(self.url, expected_path)
