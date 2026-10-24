import React from 'react';
import { PROJECT_NAME, TEAM_NAME, INSTITUTION, CHALLENGE_NAME } from '../config';

interface FooterProps {
  variant?: 'light' | 'navy';
}

export const Footer: React.FC<FooterProps> = ({ variant = 'light' }) => {
  if (variant === 'navy') {
    return (
      <footer className="bg-[#1B1E4A] text-[#C9CEF0] border-t border-[#2A2E63] py-12 px-4 sm:px-6">
        <div className="max-w-[1160px] mx-auto grid grid-cols-1 md:grid-cols-3 gap-8 text-xs font-sans">
          <div>
            <div className="text-white text-base font-bold mb-2 font-sans">{PROJECT_NAME}</div>
            <div className="text-[#AEB4E0] font-mono mb-1">{TEAM_NAME} · {INSTITUTION}</div>
            <div className="text-[#8E95C7]">{CHALLENGE_NAME}</div>
          </div>
          <div>
            <p className="leading-relaxed text-[#AEB4E0]">
              Anonymised public data only, in line with the DPDP Act and HIPAA. Decision support, not a medical device.
            </p>
          </div>
          <div>
            <p className="leading-relaxed text-[#AEB4E0] mb-2">
              Data: Zhao et al., Scientific Data, 2023 (CC BY 4.0). Code: MIT licence.
            </p>
            <span className="text-white font-mono underline">
              Source code on GitHub
            </span>
          </div>
        </div>
      </footer>
    );
  }

  return (
    <footer className="w-full border-t border-[#E3E6F1] py-8 px-4 sm:px-6 mt-16 bg-white">
      <div className="max-w-[1160px] mx-auto text-xs font-mono text-[#4A4F73]">
        Decision support, not a medical device. Anonymised public data (ShanghaiT2DM, CC BY 4.0). {TEAM_NAME} · {INSTITUTION}.
      </div>
    </footer>
  );
};
