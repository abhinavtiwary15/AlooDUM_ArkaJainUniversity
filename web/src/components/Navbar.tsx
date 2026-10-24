import React from 'react';
import { PROJECT_NAME } from '../config';

interface NavbarProps {
  currentTab: 'home' | 'triage' | 'patient' | 'evidence';
  onSelectTab: (tab: 'home' | 'triage' | 'patient' | 'evidence') => void;
  selectedPatientId?: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  onSelectTab,
}) => {
  const navItems: Array<{ id: 'home' | 'triage' | 'patient' | 'evidence'; label: string }> = [
    { id: 'home', label: 'Home' },
    { id: 'triage', label: 'Triage' },
    { id: 'patient', label: 'Patient' },
    { id: 'evidence', label: 'Evidence' },
  ];

  return (
    <header className="bg-[#1B1E4A] text-white w-full sticky top-0 z-50 border-b border-[#2A2E63]">
      <div className="max-w-[1160px] mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand / Logo */}
        <div className="flex items-center space-x-8">
          <button
            onClick={() => onSelectTab('home')}
            className="flex items-center space-x-2.5 text-left focus-visible:outline-white"
          >
            {/* Logo Mark: matching the minimal sparkline mark from screenshots */}
            <div className="w-6 h-6 bg-white text-[#1B1E4A] flex items-center justify-center rounded-[2px] font-mono text-xs font-bold leading-none select-none">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <rect x="2" y="3" width="2.5" height="10" rx="0.5" />
                <circle cx="8" cy="11.5" r="1.3" />
                <circle cx="12" cy="7.5" r="1.3" />
              </svg>
            </div>
            <span className="text-xl font-bold tracking-tight text-white font-sans">
              {PROJECT_NAME}
            </span>
          </button>
        </div>

        {/* Center / Right Links */}
        <div className="flex items-center space-x-6 sm:space-x-8">
          <nav className="flex items-center space-x-6 sm:space-x-8 text-sm font-medium">
            {navItems.map((item) => {
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`py-1.5 transition-colors relative focus-visible:outline-white ${
                    isActive
                      ? 'text-white font-semibold'
                      : 'text-[#C9CEF0] hover:text-white'
                  }`}
                >
                  {item.label}
                  {isActive && (
                    <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-white rounded-none" />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Replay mode indicator */}
          <div className="hidden sm:flex items-center space-x-2 pl-4 text-xs font-mono text-[#C9CEF0]">
            <span className="w-2 h-2 rounded-full bg-[#4ADE9C] inline-block" />
            <span>Replay mode</span>
          </div>
        </div>
      </div>
    </header>
  );
};
