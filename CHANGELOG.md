# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **SLA Management**: Implemented automatic SLA (Service Level Agreement) calculation per sector
  - Added `sla_por_setor()` method to Sinistro model that returns SLA days and label for each sector
  - SLA rules by sector:
    - CLIENTE: 5 dias corridos
    - PRECIFICAÇÃO: 5 dias corridos
    - FINANCEIRO: 4 dias corridos
    - DESMOBILIZAÇÃO / MEDIÇÃO: sem prazo
    - DEPTO.SINISTRO: 60 dias corridos
    - MANUTENÇÃO: 15 dias corridos (2 dias if aguarda_aprovacao_os is True)
    - FINALIZADO: sem prazo
    - ABERTURA: sem prazo

- **Automatic retornar_ate Calculation**: System now automatically sets the `retornar_ate` field when:
  - Sector is changed
  - `aguarda_aprovacao_os` flag is toggled
  - `retornar_ate` field is empty
  - Calculation uses current date + SLA days (calendar days)

- **Enhanced UI for SLA Display**:
  - Added dynamic SLA label display next to sector selection in edit form
  - Label updates automatically when sector changes or aguarda_aprovacao_os is toggled
  - Shows "SLA máximo: X dias corridos" or "SEM PRAZO" as appropriate

- **New Sectors**: Added missing sectors to SETORES choices:
  - PRECIFICACAO (Precificação)
  - DESMOBILIZACAO (Desmobilização / Medição)
  - DEPTO_SINISTRO (Depto. Sinistro)

- **Comprehensive Test Suite**: Created `sinistros/tests/test_aprovador_sla.py` with tests covering:
  - SLA calculation for all sectors
  - Special handling for MANUTENCAO sector with/without approval
  - Field persistence for aprovador_os
  - Automatic retornar_ate calculation logic

### Changed

- Updated SETORES field choices to include new sectors
- Enhanced editar_sinistro_view to automatically calculate and set retornar_ate based on SLA
- Improved JavaScript in editar_sinistro.html template to dynamically update SLA display

### Technical Details

- Field `aprovador_os` already exists in model (max_length=255, nullable)
- Field `aguarda_aprovacao_os` already exists in model (BooleanField)
- SLA calculation considers the aguarda_aprovacao_os flag for MANUTENCAO sector
- All changes maintain backward compatibility with existing data
