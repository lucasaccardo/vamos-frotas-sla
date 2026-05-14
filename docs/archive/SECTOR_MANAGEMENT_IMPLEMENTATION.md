# Sector Management Updates - Implementation Summary

## Overview
This implementation adds sector renaming, a new predefined sector, and custom sector creation functionality to the Vamos Frotas SLA system.

## Changes Implemented

### 1. Sector Renaming
- **Changed**: 'Manutenção' → 'Manutenção (Aprovação de O.S)'
- **Location**: `sinistros/models.py`, line 14
- **Impact**: All references to this sector now display the updated name

### 2. New Predefined Sector
- **Added**: 'MANUTENCAO_CRIACAO' with display name 'Manutenção (Criação de Processo)'
- **Location**: `sinistros/models.py`, line 15
- **SLA**: 10 days (configurable)
- **Impact**: New option available in all sector dropdowns

### 3. Custom Sector Creation (Admin Only)

#### New Model: CustomSetor
- **File**: `sinistros/models.py`, lines 191-208
- **Fields**:
  - `key`: Unique identifier (e.g., "CUSTOM_SETOR_1")
  - `display_name`: User-friendly name displayed in UI
  - `sla_days`: Optional SLA in days (null = no SLA)
  - `is_active`: Enable/disable sector
  - `created_by`: User who created the sector
  - `created_at`: Creation timestamp

#### Form: CustomSetorForm
- **File**: `sinistros/forms.py`, lines 99-134
- **Validation**:
  - Key must be alphanumeric with underscores only
  - Key must not conflict with predefined sectors
  - Key must be unique among custom sectors
  - SLA days must be positive integer or null

#### Views
- **criar_setor_customizado_view**: Create new custom sectors (admin only)
  - Location: `sinistros/views.py`, lines 1025-1044
  - Requires: `@user_passes_test(lambda u: u.is_staff)`
  - Redirects back to form after creation to show new sector in list
  
- **desativar_setor_customizado_view**: Deactivate custom sectors (admin only)
  - Location: `sinistros/views.py`, lines 1047-1057
  - Requires: `@user_passes_test(lambda u: u.is_staff)`
  - Sets `is_active=False` instead of deleting

#### Templates
- **criar_setor_customizado.html**: Admin interface for managing custom sectors
  - Form to create new sectors with key, name, and SLA
  - Table showing all existing custom sectors
  - Deactivate button for each active sector
  - Location: `templates/sinistros/criar_setor_customizado.html`

- **editar_sinistro.html**: Updated to show "Create new sector" link for admins
  - Link appears below sector dropdown
  - Only visible to users with `is_staff=True`
  - Opens in new tab
  - Location: `templates/sinistros/editar_sinistro.html`, lines 96-102

### 4. Dynamic Sector Choices

#### Helper Function: get_all_sector_choices()
- **Location**: `sinistros/views.py`, lines 28-43
- **Purpose**: Merges predefined and active custom sectors
- **Returns**: List of (key, display_name) tuples
- **Used by**: 
  - `sinistros_home_view` (filter dropdown)
  - `novo_sinistro_view` (form initialization)
  - `editar_sinistro_view` (form initialization)

#### Updated Views
- **sinistros_home_view**: Uses `get_all_sector_choices()` for filter dropdown
- **novo_sinistro_view**: Dynamically sets form choices on both GET and POST
- **editar_sinistro_view**: Dynamically sets form choices on both GET and POST

### 5. SLA Logic Updates

#### Model Method: sla_por_setor()
- **Location**: `sinistros/models.py`, lines 120-176
- **Changes**:
  - Added special case for MANUTENCAO_CRIACAO (10 days)
  - Added custom sector lookup for SLA calculation
  - Falls back to predefined rules if not found

#### Template Updates
- **editar_sinistro.html**: 
  - Added `data-manutencao_criacao="10 dias corridos"` to SLA config
  - Added `MANUTENCAO_CRIACAO` to JavaScript SLA_MAP
  - Location: lines 52, 277

## Database Migrations

### Migration 0008
- **File**: `sinistros/migrations/0008_alter_sinistro_setor_atual_customsetor.py`
- **Changes**:
  - Altered `setor_atual` choices to include renamed sector
  - Created `CustomSetor` table

### Migration 0009
- **File**: `sinistros/migrations/0009_customsetor_sla_days.py`
- **Changes**:
  - Added `sla_days` field to `CustomSetor` model

## URL Patterns

Added to `sinistros/urls.py`:
```python
path('admin/criar-setor/', views.criar_setor_customizado_view, name='criar_setor_customizado'),
path('admin/desativar-setor/<int:sector_id>/', views.desativar_setor_customizado_view, name='desativar_setor_customizado'),
```

## Security

### Access Control
- Custom sector creation requires `is_staff=True`
- Enforced via `@user_passes_test(lambda u: u.is_staff)` decorator
- Link to create sectors only shown to staff users in templates

### Validation
- Key format validation (alphanumeric + underscores only)
- Uniqueness validation for custom sector keys
- Conflict detection with predefined sector keys
- CodeQL security scan: **0 alerts**

## Testing

### Existing Tests
- All 40 existing tests pass without modification
- Tests cover:
  - SLA calculations for all sectors
  - Approval workflow
  - Data export functionality
  - Sector filtering

### Manual Verification
Created `/tmp/test_sectors.py` to verify:
- ✓ Sector renaming displays correctly
- ✓ New sector appears in choices
- ✓ Custom sector model works correctly
- ✓ Custom sectors appear in `get_all_sector_choices()`
- ✓ SLA logic works for custom sectors

## Usage Instructions

### For Admin Users

#### Creating a Custom Sector
1. Navigate to a sinistro edit page
2. Click "Criar novo setor (Admin)" link below sector dropdown
3. Fill in:
   - **Chave do Setor**: Unique key (e.g., "MANUTENCAO_ESPECIAL")
   - **Nome do Setor**: Display name (e.g., "Manutenção Especial")
   - **Prazo SLA (dias)**: Optional SLA in days (leave empty for no SLA)
4. Click "Criar Setor"
5. Sector immediately appears in all sector dropdowns

#### Deactivating a Custom Sector
1. Go to custom sector creation page
2. Find sector in table
3. Click "Desativar" button
4. Sector no longer appears in dropdowns but remains in database

### For Regular Users
- See renamed sector in all dropdowns
- See new "Manutenção (Criação de Processo)" option
- See any active custom sectors created by admins
- Cannot create or deactivate custom sectors

## Files Modified

1. `sinistros/models.py` - Model changes
2. `sinistros/forms.py` - Form additions
3. `sinistros/views.py` - View updates and new views
4. `sinistros/urls.py` - URL patterns
5. `templates/sinistros/editar_sinistro.html` - Admin link and SLA config
6. `templates/sinistros/criar_setor_customizado.html` - New template

## Files Created

1. `sinistros/migrations/0008_alter_sinistro_setor_atual_customsetor.py`
2. `sinistros/migrations/0009_customsetor_sla_days.py`
3. `templates/sinistros/criar_setor_customizado.html`

## Backward Compatibility

- All existing sectors continue to work
- Existing sinistros with "MANUTENCAO" sector automatically display new name
- No data migration required
- All existing tests pass without modification

## Future Enhancements (Not Implemented)

- Dashboard stats for custom sectors
- Bulk import of custom sectors
- Sector usage analytics
- Historical sector name tracking
- Custom sector permissions per user group
