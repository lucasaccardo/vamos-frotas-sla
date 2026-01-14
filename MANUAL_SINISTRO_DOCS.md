# Manual de Sinistro - Interactive Flowchart Implementation

## Overview
This document describes the implementation of the "Manual de Sinistro" (Claims Manual) interactive flowchart feature integrated into the Gestão de Manutenção (Maintenance Management) section.

## Architecture

### Technology Stack
- **Frontend**: React 18 (via CDN) with functional components and hooks
- **Styling**: Tailwind CSS (via CDN) for modern, responsive design
- **JSX Transpilation**: Babel Standalone for in-browser JSX transformation
- **Backend**: Django (Python) for routing and authentication
- **Data**: JSON-based workflow definition for branching logic

### Design Principles
1. **Zero Manual Input**: Complete click-based navigation system
2. **JSON-Driven**: All workflow logic defined in external JSON file
3. **Responsive**: Mobile-first design with Tailwind CSS
4. **Accessible**: Semantic HTML with proper ARIA labels
5. **Animated**: Smooth transitions for better UX

## File Structure

```
/vamos-frotas-sla/
├── static/
│   └── data/
│       └── manual-sinistro-flow.json     # Workflow definition
├── vamos/
│   ├── templates/
│   │   └── vamos/
│   │       ├── base.html                  # Modified: Added sidebar link
│   │       ├── home.html                  # Modified: Added feature card
│   │       └── manual_sinistro.html       # New: React SPA template
│   ├── urls.py                            # Modified: Added route
│   └── views.py                           # Modified: Added view
```

## JSON Workflow Structure

The workflow is defined in `static/data/manual-sinistro-flow.json`:

```json
{
  "flowName": "Manual de Sinistro - Fluxo Interativo",
  "version": "1.0",
  "startNodeId": "inicio",
  "nodes": {
    "node_id": {
      "id": "node_id",
      "type": "question|information|checklist|email|completion",
      "title": "Node Title",
      "description": "Node description",
      "icon": "🚗",
      "content": [...],         // For information/completion nodes
      "items": [...],           // For checklist nodes
      "emailTo": "...",         // For email nodes
      "emailSubject": "...",    // For email nodes
      "emailBody": "...",       // For email nodes
      "options": [
        {
          "label": "Option text",
          "nextNode": "next_node_id",
          "color": "primary|success|danger|warning|secondary|info"
        }
      ]
    }
  }
}
```

### Node Types

1. **question**: Presents multiple options to the user
2. **information**: Displays informational content with list items
3. **checklist**: Shows a checklist of required documents
4. **email**: Generates an email template with copy-to-clipboard functionality
5. **completion**: Final confirmation screen

## React Components

### ManualSinistroApp (Main Component)
- **State Management**:
  - `flowData`: Complete workflow loaded from JSON
  - `currentNodeId`: Current position in the workflow
  - `history`: Array of visited node IDs for back navigation
  - `loading`: Loading state indicator
  - `copiedEmail`: Feedback for clipboard copy action

- **Functions**:
  - `navigateToNode(nodeId)`: Move to a specific node
  - `goBack()`: Navigate to previous node
  - `resetFlow()`: Reset to start
  - `copyToClipboard(text)`: Copy email to clipboard

### FlowNode (Display Component)
- Renders different UI based on node type
- Handles button styling based on color prop
- Manages animations for smooth transitions

## Key Features

### 1. Progress Tracking
- Visual progress bar showing completion percentage
- Calculated based on nodes visited vs total nodes
- Smooth animation on progress updates

### 2. Navigation
- **Forward**: Click option buttons to proceed
- **Backward**: Click "Voltar" (Back) button
- **Reset**: Click "Reiniciar" (Restart) button at any time

### 3. Email Generation
- Formats email templates with placeholders
- One-click copy to clipboard
- Visual feedback ("Copiado!" message)
- Pre-filled To, Subject, and Body

### 4. Animations
- **Fade-in**: Main cards and containers
- **Slide-in**: Content within cards
- **Hover effects**: Button hover with translate and shadow
- **Progress bar**: Smooth width transition

## Workflow Examples

### Example 1: Collision with Third Party
```
início → tipo_sinistro → colisao_terceiros → docs_colisao → email_colisao → conclusao
```

### Example 2: Vehicle Theft
```
início → tipo_sinistro → furto_roubo → docs_furto → email_furto → conclusao
```

## Django Integration

### URL Configuration (`vamos/urls.py`)
```python
path("manual-sinistro/", views.manual_sinistro_view, name="manual_sinistro"),
```

### View (`vamos/views.py`)
```python
@login_required(login_url='login')
def manual_sinistro_view(request):
    """
    Exibe o Manual de Sinistro interativo com fluxograma de decisão.
    Sistema de navegação baseado em cliques com lógica de ramificação em JSON.
    """
    return render(request, 'vamos/manual_sinistro.html')
```

### Template (`vamos/templates/vamos/manual_sinistro.html`)
- Extends `vamos/base.html` for consistent layout
- Loads React and Tailwind CSS from CDN
- Uses `{% verbatim %}` tag to prevent Django template parsing of JSX
- Embeds complete React application in single template

## Customization Guide

### Adding New Nodes
1. Open `static/data/manual-sinistro-flow.json`
2. Add new node to `nodes` object:
```json
"new_node_id": {
  "id": "new_node_id",
  "type": "information",
  "title": "New Step Title",
  "description": "Description text",
  "icon": "🎯",
  "content": ["Item 1", "Item 2"],
  "options": [
    {
      "label": "Continue",
      "nextNode": "next_step_id",
      "color": "primary"
    }
  ]
}
```

### Modifying Email Templates
1. Locate the email node in JSON (e.g., `email_colisao`)
2. Update `emailBody` field with new template
3. Use `[PLACEHOLDERS]` for user-fillable fields
4. Maintain markdown-style formatting (`**bold**`, line breaks)

### Changing Styles
1. **Colors**: Modify Tailwind classes in template
2. **Animations**: Update keyframes in `<style>` section
3. **Layout**: Adjust Tailwind utility classes in JSX

## Browser Compatibility
- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support (iOS 12+)
- IE11: ❌ Not supported (requires modern browser)

## Performance Considerations
- **CDN Loading**: React and Tailwind loaded from CDN (cached globally)
- **JSON Size**: ~18KB (minimal impact on load time)
- **Babel Transpilation**: In-browser (slight delay on initial load)
- **Optimization**: Consider building static bundle for production

## Security Features
- **Authentication Required**: `@login_required` decorator on view
- **CSRF Protection**: Django's built-in CSRF middleware
- **No User Input**: Click-only interface eliminates injection risks
- **Static Content**: JSON served as static file

## Accessibility
- Semantic HTML structure
- High contrast colors (WCAG AA compliant)
- Keyboard navigation support
- Screen reader compatible
- Focus indicators on interactive elements

## Testing

### Manual Testing Checklist
- [ ] Load page and verify no errors in console
- [ ] Navigate through complete workflow
- [ ] Test back button functionality
- [ ] Test reset button
- [ ] Copy email to clipboard
- [ ] Verify mobile responsiveness
- [ ] Test different node types
- [ ] Verify progress bar updates
- [ ] Test on multiple browsers

### Automated Testing
```python
# Test view access
from django.test import Client
from django.contrib.auth.models import User

user = User.objects.get(username='testuser')
client = Client()
client.force_login(user)
response = client.get('/manual-sinistro/')
assert response.status_code == 200
assert b'Manual de Sinistro' in response.content
```

## Future Enhancements
1. **Backend Integration**: Save user progress to database
2. **Analytics**: Track most common paths through workflow
3. **Multi-language**: Support for English/Spanish versions
4. **File Uploads**: Allow users to attach documents
5. **PDF Generation**: Export completed workflow as PDF
6. **Email Integration**: Send emails directly from interface
7. **Admin Panel**: GUI for editing JSON workflow
8. **Production Build**: Pre-compile JSX and host libraries locally for improved performance
9. **SRI Hashes**: Add Subresource Integrity hashes to CDN links for enhanced security
10. **Local Assets**: Host React and Tailwind locally instead of CDN for production

## Security Considerations

### CDN Usage
The current implementation uses CDN-hosted libraries (React, Tailwind CSS, Babel) for simplicity and ease of deployment. While suitable for internal applications, consider the following for production:

**Advantages:**
- ✅ No build process required
- ✅ Globally cached (faster load times)
- ✅ Easy to update
- ✅ Minimal setup

**Considerations:**
- ⚠️ CDN availability dependency
- ⚠️ Potential security risk if CDN compromised
- ⚠️ Network latency for first load

**Production Recommendations:**
1. Add Subresource Integrity (SRI) hashes to CDN script tags
2. Implement fallback to local copies if CDN fails
3. Consider hosting libraries locally for critical production use
4. Use a build process (Webpack/Vite) to bundle and optimize assets

### Example with SRI:
```html
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js" 
        integrity="sha384-..." crossorigin="anonymous"></script>
```

## Troubleshooting

### Issue: Template Syntax Error
**Cause**: Django trying to parse JSX curly braces
**Solution**: Ensure JSX is wrapped in `{% verbatim %}` tags

### Issue: JSON Not Loading
**Cause**: Incorrect static file path or collectstatic not run
**Solution**: Run `python manage.py collectstatic` and verify path

### Issue: React Not Rendering
**Cause**: CDN blocked or JavaScript disabled
**Solution**: Check browser console for CDN errors, enable JavaScript

### Issue: Sidebar Link Not Showing
**Cause**: Wrong module active in session
**Solution**: Verify `request.session.modulo_ativo == 'manutencao'`

## Maintenance

### Updating Workflow
1. Backup current JSON file
2. Edit `manual-sinistro-flow.json`
3. Test in development environment
4. Run collectstatic to deploy
5. Clear browser cache for users

### Version Control
- Track JSON versions in `version` field
- Document changes in CHANGELOG
- Consider semantic versioning (1.0, 1.1, 2.0)

## Support
For questions or issues:
1. Check this documentation first
2. Review Django logs for errors
3. Check browser console for JavaScript errors
4. Contact development team

## License
Internal use only - Vamos Frotas SLA System

---

**Last Updated**: 2026-01-14
**Version**: 1.0
**Author**: Development Team
