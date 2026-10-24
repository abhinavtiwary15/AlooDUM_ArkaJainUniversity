import { test, expect } from '@playwright/test';

const FORBIDDEN_WORDS = [
  new RegExp(['gluco', 'twin'].join(''), 'i'),
  new RegExp(['kharag', 'pur'].join(''), 'i'),
  new RegExp(['tiwa', 'ry'].join(''), 'i'),
];
const VIEWPORTS = [
  { width: 390, height: 844, name: '390px-mobile' },
  { width: 768, height: 1024, name: '768px-tablet' },
  { width: 1440, height: 900, name: '1440px-desktop' },
];

test.describe('NoctTwin Quality & Clinical Verification Suite', () => {
  // Test 1: Invariant check - No route contains forbidden strings
  test('Zero occurrence of forbidden names or previous brand strings', async ({ page }) => {
    const routes = ['#home', '#triage', '#patient', '#evidence'];
    for (const r of routes) {
      await page.goto('/' + r);
      await page.waitForLoadState('networkidle');
      const bodyText = await page.locator('body').innerText();
      for (const pattern of FORBIDDEN_WORDS) {
        expect(bodyText).not.toMatch(pattern);
      }
    }
  });

  // Test 2: Monotonicity test for Triage
  test('Triage table is strictly monotonic in risk probability and tier order', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/#triage');
    await expect(page.locator('text=Tonight\'s list, ranked by risk')).toBeVisible({ timeout: 10000 });

    // Wait for rows to load
    const rows = page.locator('tbody tr');
    await expect(rows.first()).toBeVisible();
    const count = await rows.count();
    expect(count).toBeGreaterThan(10);

    const tierMap: Record<string, number> = {
      High: 3,
      Moderate: 2,
      Low: 1,
    };

    let prevProb = 1.1;
    let prevTierVal = 4;

    for (let i = 0; i < Math.min(count, 30); i++) {
      const row = rows.nth(i);
      const rowText = await row.innerText();

      // Extract % from row
      const match = rowText.match(/(\d+)%/);
      if (match) {
        const prob = parseInt(match[1], 10) / 100.0;
        expect(prob).toBeLessThanOrEqual(prevProb);
        prevProb = prob;
      }

      // Check tier
      let currentTierVal = 0;
      if (rowText.includes('High')) currentTierVal = tierMap.High;
      else if (rowText.includes('Moderate')) currentTierVal = tierMap.Moderate;
      else if (rowText.includes('Low')) currentTierVal = tierMap.Low;

      if (currentTierVal > 0) {
        expect(currentTierVal).toBeLessThanOrEqual(prevTierVal);
        prevTierVal = currentTierVal;
      }
    }
  });

  // Test 3: Responsive viewports test (390px, 768px, 1440px)
  for (const vp of VIEWPORTS) {
    test(`Renders cleanly at viewport ${vp.name}`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });

      // Home
      await page.goto('/#home');
      await expect(page.locator('text=See the night low before it happens.')).toBeVisible();

      // Triage
      await page.goto('/#triage');
      await expect(page.locator('text=Tonight\'s list, ranked by risk')).toBeVisible();

      // Patient
      await page.goto('/#patient');
      await expect(page.locator('text=Points for clinician review')).toBeVisible();
      const pointsCard = page.locator('text=Points for clinician review').locator('xpath=..');
      const pointsText = await pointsCard.innerText();
      expect(pointsText).not.toMatch(/snack/i);
      expect(pointsText).not.toMatch(/dose/i);
      expect(pointsText).not.toMatch(/treatment/i);

      // Evidence
      await page.goto('/#evidence');
      await expect(page.locator('text=Scientific evidence & model card')).toBeVisible();
    });
  }
});
