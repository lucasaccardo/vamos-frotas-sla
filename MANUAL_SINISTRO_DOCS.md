# Manual de Sinistro - Interactive Flowchart Implementation

## Overview
This document describes the implementation of the "Manual de Sinistro" (Claims Manual) interactive flowchart feature integrated into the Gestão de Sinistros (Claims Management) section. The system implements three main payment process flows based on branching decision logic.

## Version 2.0 - Payment Process Structure
**Release Date**: 2026-01-15

This version refocuses the manual from incident types to **payment processes**, implementing three distinct workflows:

1. **Pagamento pelo Cliente** - Customer pays 100% of repair costs
2. **Pagamento pelo Cliente - Co-participação** - Customer pays a percentage + insurance covers the rest
3. **Pagamento pela Seguradora** - Insurance pays 100% of repair costs

### Key Statistics
- **Total Nodes**: 43 interconnected decision nodes
- **Email Templates**: 5 comprehensive templates
- **Question Nodes**: 14 decision points
- **Information Nodes**: 15 instructional screens
- **Completion Nodes**: 9 end/waiting states
- **FAQ Questions**: 10 payment-focused questions

## Architecture

### Technology Stack
- **Frontend**: React 18 (via CDN) with functional components and hooks
- **Styling**: Tailwind CSS (via CDN) for modern, responsive design
- **JSX Transpilation**: Babel Standalone for in-browser JSX transformation
- **Backend**: Django (Python) for routing and authentication
- **Data**: JSON-based workflow definition for branching logic (Version 2.0)

### Design Principles
1. **Zero Manual Input**: Complete click-based navigation system - no typing required
2. **JSON-Driven**: All workflow logic defined in external JSON file for easy updates
3. **Three Payment Processes**: Clear separation between customer payment, co-participation, and insurance payment
4. **Responsive**: Mobile-first design with Tailwind CSS
5. **Accessible**: Semantic HTML with proper ARIA labels
6. **Animated**: Smooth fade-in and slide-in transitions for better UX
7. **Decision Trees**: Branching logic based on user selections

## Payment Process Flows

### Flow 1: Pagamento pelo Cliente (Customer Payment)
**Description**: Customer assumes 100% of repair costs
**Nodes**: 13 nodes
**Key Features**:
- Verify if claim is registered
- Request/approve quotation
- Choose payment method (cash or installments)
- Cash: 5% discount, 3x interest-free or 6x with 1.99% interest/month
- Email templates for payment request

**Sample Path**:
```
inicio → pagamento_cliente → cliente_verificar_sinistro →
cliente_possui_orcamento → cliente_forma_pagamento →
cliente_pagamento_vista → cliente_apos_pagamento
```

### Flow 2: Pagamento com Co-participação (Co-participation)
**Description**: Customer pays percentage + insurance covers the rest
**Nodes**: 14 nodes
**Key Features**:
- Insurance analysis and approval (3-5 days)
- Quotation with defined percentage
- Calculate customer's share (typically 10-30%)
- Payment options: 2x interest-free or 4x with 2.49% interest/month
- Email templates for co-participation payment

**Sample Path**:
```
inicio → pagamento_coparticipacao → copart_verificar_sinistro →
copart_possui_orcamento → copart_calcular_valores →
copart_forma_pagamento → copart_pagamento_parcelado →
copart_apos_pagamento
```

### Flow 3: Pagamento pela Seguradora (Insurance Payment)
**Description**: Insurance pays 100% directly to repair shop
**Nodes**: 15 nodes
**Key Features**:
- Communicate claim to insurance
- Wait for coverage approval (3-7 days)
- Schedule and complete inspection
- Choose repair shop (referral or free choice)
- Insurance pays directly to shop (zero cost to customer)
- Email template for repair authorization

**Sample Path**:
```
inicio → pagamento_seguradora → seguradora_verificar_cobertura →
seguradora_numero_sinistro → seguradora_agendar_pericia →
seguradora_escolher_oficina → seguradora_enviar_veiculo →
seguradora_acompanhamento
```

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

The workflow is defined in `static/data/manual-sinistro-flow.json` (Version 2.0):

### Top-Level Structure
```json
{
  "flowName": "Manual de Sinistro - Fluxo Interativo de Pagamento",
  "version": "2.0",
  "startNodeId": "inicio",
  "nodes": {
    // 43 nodes defining the complete payment process flows
  }
}
```

### Node Structure
```json
"node_id": {
  "id": "node_id",
  "type": "question|information|checklist|email|completion",
  "title": "Node Title",
  "description": "Node description",
  "icon": "🚗",
  "content": [...],         // For information/completion nodes
  "items": [...],           // For checklist nodes (not used in v2.0)
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
```

### Node Types

1. **question**: Presents multiple options to the user (14 nodes)
2. **information**: Displays informational content with list items (15 nodes)
3. **email**: Generates an email template with copy-to-clipboard functionality (5 nodes)
4. **completion**: Final confirmation screen or waiting state (9 nodes)

## Email Templates

The system includes 5 comprehensive email templates covering all payment scenarios:

### 1. Pagamento à Vista - Instruções
- **Recipient**: financeiro@vamosfrotas.com.br
- **Use Case**: Customer requests cash payment details
- **Features**: 5% discount mention, payment method options, invoice data fields

### 2. Pagamento Parcelado - Instruções
- **Recipient**: financeiro@vamosfrotas.com.br
- **Use Case**: Customer requests installment payment
- **Features**: Installment options (3x or 6x), card data fields, invoice data

### 3. Pagamento à Vista - Co-participação
- **Recipient**: financeiro@vamosfrotas.com.br
- **Use Case**: Customer requests co-participation cash payment
- **Features**: Insurance approval data, percentage and value breakdown

### 4. Pagamento Parcelado - Co-participação
- **Recipient**: financeiro@vamosfrotas.com.br
- **Use Case**: Customer requests co-participation installment
- **Features**: 2x or 4x options, insurance approval data

### 5. Autorizar Reparo pela Seguradora
- **Recipient**: sinistros@vamosfrotas.com.br
- **Use Case**: Authorize repair with insurance payment
- **Features**: Insurance data, inspection details, repair shop info, timeline

## FAQ Section

The system includes 10 payment-focused FAQ questions:

1. **Payment Types**: Differences between the three payment processes
2. **Determining Payment Type**: How to identify which applies
3. **Installment Options**: Details on payment plans available
4. **Processing Time**: Timelines for each payment type
5. **Mandatory Deductible**: Explanation of insurance deductibles
6. **Repair Shop Selection**: Options for each payment type
7. **Vehicle Use During Processing**: Safety and coverage considerations
8. **Insurance Denial**: Options when insurance refuses coverage
9. **Non-payment Consequences**: What happens if co-participation isn't paid
10. **Changing Payment Method**: Possibility of switching between processes

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
