/**
 * NoctTwin System Configuration & Design Tokens
 */

export const PROJECT_NAME = 'NoctTwin';
export const TEAM_NAME = 'Team AlooDUM';
export const INSTITUTION = 'Arka Jain University';
export const CHALLENGE_NAME = 'Happiest Health Digital Twin Challenge 2026';

export const COLORS = {
  navy: '#1B1E4A',
  ink: '#0E1330',
  bodyText: '#3A3F66',
  secondaryText: '#4A4F73',
  panel: '#F1F3F9',
  panelHover: '#E8ECF5',
  hairline: '#E3E6F1',
  hairlineDark: '#CBD0E2',
  white: '#FFFFFF',
  emerald: '#046A38',
  emeraldBg: '#E2F1E8',
  mint: '#4ADE9C',
  crimson: '#D21319',
  crimsonBg: '#FCE4E4',
  crimsonNavy: '#FF7A7F',
  ochre: '#8A5A00',
  ochreBg: '#FFF3D6',
  ochreNavy: '#F2B84B',
  navyMuted: '#AEB4E0',
  navyLight: '#C9CEF0',
};

export const TIERS = {
  HIGH: {
    label: 'High',
    symbol: '▲',
    color: '#A30F14',
    bg: '#FCE4E4',
    border: '#F8B4B4',
  },
  MODERATE: {
    label: 'Moderate',
    symbol: '◆',
    color: '#7A4F00',
    bg: '#FFF3D6',
    border: '#FFE194',
  },
  LOW: {
    label: 'Low',
    symbol: '●',
    color: '#046A38',
    bg: '#E2F1E8',
    border: '#BDE3CC',
  },
};

export const THRESHOLDS = {
  hypo: 70, // mg/dL
  severeHypo: 54, // mg/dL
  anchorHour: 22, // 22:00
};
