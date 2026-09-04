import crypto from "node:crypto";
import http from "node:http";

const SCHEMA = "neutralgrid_extension_bridge_v1";
const SYMBOL_RE = /^[A-Z0-9]+USDT$/;
const STRATEGY_RE = /^\d+$/;
const TOKEN_RE = /^[0-9a-f]{64}$/;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function safeEqual(left, right) {
  const a = Buffer.from(String(left));
  const b = Buffer.from(String(right));
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

async function poll(tab, predicate, label, timeoutMs = 8000) {
  const expires = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < expires) {
    try {
      if (await predicate()) return;
    } catch (error) {
      lastError = error;
    }
    await tab.playwright.waitForTimeout(100);
  }
  const suffix = lastError ? `: ${lastError.message || lastError}` : "";
  throw new Error(`${label} timed out${suffix}`);
}

async function uniqueVisible(locator, label) {
  const count = await locator.count();
  const visible = [];
  for (let index = 0; index < count; index += 1) {
    const candidate = locator.nth(index);
    if (await candidate.isVisible()) visible.push(candidate);
  }
  assert(visible.length === 1, `${label} count=${visible.length}`);
  return visible[0];
}

function validateIdentity(payload) {
  const symbol = String(payload.symbol || "").toUpperCase();
  const strategyId = String(payload.strategy_id || "");
  assert(SYMBOL_RE.test(symbol), "invalid symbol");
  assert(STRATEGY_RE.test(strategyId), "invalid strategy_id");
  return { symbol, strategyId };
}

async function closeDrawer(tab) {
  const drawer = tab.playwright.getByRole("dialog", { name: "drawer" });
  if (!(await drawer.isVisible())) return;
  const closes = tab.playwright.locator("div.fixed.top-0.right-0 > div > svg");
  assert((await closes.count()) === 1, "drawer close control is unavailable");
  const close = closes.first();
  assert(await close.isVisible(), "drawer close control is hidden");
  await close.click();
  await poll(tab, async () => !(await drawer.isVisible()), "drawer close");
}

async function ensureSymbolPage(tab, symbol) {
  const target = `https://www.binance.bh/en/trading-bots/futures/grid/${symbol}`;
  if ((await tab.url()) !== target) await tab.goto(target);
  await poll(
    tab,
    async () => {
      if ((await tab.url()) !== target) return false;
      const umGrid = tab.playwright.getByRole("tab").filter({ hasText: "UM Grid" });
      return (await umGrid.count()) === 1 && (await umGrid.isVisible());
    },
    "exact symbol page load",
    20000,
  );
}

async function openExactDrawer(tab, payload) {
  const { symbol, strategyId } = validateIdentity(payload);
  await ensureSymbolPage(tab, symbol);
  await closeDrawer(tab);
  const row = tab.playwright
    .getByRole("row")
    .filter({ hasText: `${symbol} Perp` })
    .filter({ hasText: "Working" });
  await poll(
    tab,
    async () => (await row.count()) === 1 && (await row.isVisible()),
    `${symbol} Working row render`,
    10000,
  );
  assert((await row.count()) === 1, `expected one Working row for ${symbol}`);
  assert(await row.isVisible(), `${symbol} Working row is not visible`);
  const cells = row.getByRole("cell");
  let cellCount = 0;
  await poll(
    tab,
    async () => {
      cellCount = await cells.count();
      if (cellCount < 1) return false;
      return (await cells.nth(cellCount - 1).getByRole("button").count()) === 4;
    },
    `${symbol} action controls render`,
    10000,
  );
  assert(cellCount >= 1, `${symbol} action cell is unavailable`);
  const actionCell = cells.nth(cellCount - 1);
  const buttons = actionCell.getByRole("button");
  assert((await buttons.count()) === 4, `${symbol} action controls are ambiguous`);
  const viewDetails = buttons.nth(2);
  await viewDetails.click();
  const drawer = tab.playwright.getByRole("dialog", { name: "drawer" });
  await poll(tab, () => drawer.isVisible(), "View Details drawer open", 20000);
  let rawText = await drawer.innerText();
  if (!rawText.includes("Order History")) {
    assert(
      rawText.includes("History") && rawText.includes("Total Matched Profit"),
      "drawer history section is incomplete",
    );
    rawText += "\nOrder History";
  }
  assert(rawText.includes(symbol), "drawer symbol mismatch");
  assert(rawText.includes(strategyId), "drawer strategy mismatch");
  assert(/(?:^|\n)Working(?:\n|$)/.test(rawText), "bot is not Working");
  return rawText;
}

async function mainModal(tab) {
  return uniqueVisible(
    tab.playwright.locator(".bn-modal-wrap.data-size-medium"),
    "Modify Parameters modal",
  );
}

async function readForm(tab) {
  const modal = await mainModal(tab);
  const inputValue = async (selector, label) => {
    const input = modal.locator(selector);
    assert((await input.count()) === 1, `${label} input count mismatch`);
    const value = await input.getAttribute("value", { timeoutMs: 15000 });
    assert(value !== null, `${label} input value is unavailable`);
    return value;
  };
  const lower = await inputValue('input[placeholder="Lower"]', "lower");
  const upper = await inputValue('input[placeholder="Upper"]', "upper");
  const grids = await inputValue('input[placeholder^="2-"]', "grids");
  const text = await modal.innerText();
  const closeMatch = text.match(/Close Your Current Positions\s+(Yes|No)(?:\s|$)/);
  const investmentMatch = text.match(/Additional Investment\s+([0-9,.]+)\s+USDT/);
  const confirms = modal.getByRole("button", { name: "Confirm", exact: true });
  const confirmCount = await confirms.count();
  const confirmEnabled =
    confirmCount === 1 &&
    (await confirms.first().isVisible()) &&
    (await confirms.first().isEnabled());
  return {
    lower: String(lower),
    upper: String(upper),
    grids: String(grids),
    close_positions: closeMatch ? closeMatch[1] : null,
    additional_investment: investmentMatch ? investmentMatch[1] : null,
    confirm_enabled: confirmEnabled,
  };
}

async function openModifyForm(tab, payload) {
  const { symbol, strategyId } = validateIdentity(payload);
  const drawer = tab.playwright.getByRole("dialog", { name: "drawer" });
  assert(await drawer.isVisible(), "View Details drawer is not open");
  const rawText = await drawer.innerText();
  assert(rawText.includes(symbol) && rawText.includes(strategyId), "drawer identity changed");
  const modify = drawer.getByRole("button", { name: "Modify Parameters", exact: true });
  assert((await modify.count()) === 1, "Modify Parameters control is ambiguous");
  await modify.click();
  await poll(
    tab,
    async () => {
      try {
        await mainModal(tab);
        return true;
      } catch {
        return false;
      }
    },
    "Modify Parameters modal open",
  );
  return readForm(tab);
}

async function ensureKeepPosition(tab) {
  let form = await readForm(tab);
  if (form.close_positions === "No") return form;
  assert(form.close_positions === "Yes", "current-position choice is not parseable");
  const modal = await mainModal(tab);
  const yes = modal.getByText("Yes", { exact: true });
  assert((await yes.count()) === 1, "current-position selector is ambiguous");
  await yes.click();
  const choiceLocator = tab.playwright.locator(".bn-modal-wrap");
  let choiceModal;
  await poll(
    tab,
    async () => {
      const count = await choiceLocator.count();
      const matches = [];
      for (let index = 0; index < count; index += 1) {
        const candidate = choiceLocator.nth(index);
        if (!(await candidate.isVisible())) continue;
        if ((await candidate.innerText()).includes("No, please keep my positions")) {
          matches.push(candidate);
        }
      }
      if (matches.length === 1) {
        choiceModal = matches[0];
        return true;
      }
      return false;
    },
    "position-choice modal open",
  );
  assert(choiceModal, "position-choice modal is unavailable");
  const keep = choiceModal
    .getByRole("radio")
    .filter({ hasText: "No, please keep my positions" });
  assert((await keep.count()) === 1, "keep-position option is ambiguous");
  await keep.click();
  const confirm = choiceModal.getByRole("button", { name: "Confirm", exact: true });
  assert((await confirm.count()) === 1, "position-choice confirm is ambiguous");
  await confirm.click();
  await poll(
    tab,
    async () => (await readForm(tab)).close_positions === "No",
    "position preservation selection",
  );
  form = await readForm(tab);
  return form;
}

async function setFormInputs(tab, payload) {
  const lower = String(payload.lower || "");
  const upper = String(payload.upper || "");
  assert(/^\d+(?:\.\d+)?$/.test(lower), "invalid lower input");
  assert(/^\d+(?:\.\d+)?$/.test(upper), "invalid upper input");
  const modal = await mainModal(tab);
  const lowerInput = modal.locator('input[placeholder="Lower"]');
  const upperInput = modal.locator('input[placeholder="Upper"]');
  assert((await lowerInput.count()) === 1, "lower input is ambiguous");
  assert((await upperInput.count()) === 1, "upper input is ambiguous");
  await lowerInput.fill(lower);
  await upperInput.fill(upper);
  await poll(
    tab,
    async () => {
      const form = await readForm(tab);
      return form.lower === lower && form.upper === upper && form.confirm_enabled;
    },
    "prepared form validation",
  );
  return readForm(tab);
}

async function submit(tab, payload) {
  const expected = {
    lower: String(payload.lower || ""),
    upper: String(payload.upper || ""),
    grids: String(payload.grids || ""),
  };
  assert(payload.preserve_current_position === true, "position preservation missing");
  assert(String(payload.additional_investment) === "0", "additional investment is not zero");
  const form = await readForm(tab);
  assert(form.lower === expected.lower, "final lower input changed");
  assert(form.upper === expected.upper, "final upper input changed");
  assert(form.grids === expected.grids, "final grid count changed");
  assert(form.close_positions === "No", "final form would close the position");
  assert(["0", "0.0", "0.00"].includes(String(form.additional_investment)), "final form requires investment");
  assert(form.confirm_enabled, "final confirm is disabled");
  const modal = await mainModal(tab);
  const confirm = modal.getByRole("button", { name: "Confirm", exact: true });
  assert((await confirm.count()) === 1, "final confirm is ambiguous");
  await confirm.click();

  // Binance may insert a second, explicit "Confirm Modification" modal
  // after the prepared-form confirmation.  Treat that as the sole final
  // submit control; legacy one-stage flows remain supported when the
  // prepared-form modal closes without presenting it.
  const confirmationLocator = tab.playwright.locator(".bn-modal-wrap").filter({
    hasText: "Do you want to continue modifying the parameters?",
  });
  let confirmationModal;
  await poll(
    tab,
    async () => {
      const count = await confirmationLocator.count();
      const visible = [];
      for (let index = 0; index < count; index += 1) {
        const candidate = confirmationLocator.nth(index);
        if (await candidate.isVisible()) visible.push(candidate);
      }
      assert(visible.length <= 1, "Confirm Modification modal is ambiguous");
      if (visible.length === 1) {
        confirmationModal = visible[0];
        return true;
      }
      try {
        await mainModal(tab);
        return false;
      } catch {
        return true;
      }
    },
    "submit confirmation transition",
    12000,
  );
  if (confirmationModal) {
    const finalConfirm = confirmationModal.getByRole("button", {
      name: "Confirm",
      exact: true,
    });
    assert((await finalConfirm.count()) === 1, "final submit confirm is ambiguous");
    assert(await finalConfirm.isEnabled(), "final submit confirm is disabled");
    await finalConfirm.click();
  }
  await poll(
    tab,
    async () => {
      const confirmationCount = await confirmationLocator.count();
      for (let index = 0; index < confirmationCount; index += 1) {
        if (await confirmationLocator.nth(index).isVisible()) return false;
      }
      try {
        await mainModal(tab);
        return false;
      } catch {
        return true;
      }
    },
    "final submit acknowledgement",
    12000,
  );
  return { acknowledged: true };
}

async function dispatch(tab, action, payload) {
  switch (action) {
    case "hello":
      return { provider: "chrome-extension", url: await tab.url() };
    case "read_state":
      return { raw_text: await openExactDrawer(tab, payload) };
    case "open_modify_form":
      return openModifyForm(tab, payload);
    case "ensure_keep_position":
      return ensureKeepPosition(tab);
    case "set_form_inputs":
      return setFormInputs(tab, payload);
    case "read_form":
      return readForm(tab);
    case "submit":
      return submit(tab, payload);
    case "wait": {
      const seconds = Number(payload.seconds);
      assert(Number.isFinite(seconds) && seconds >= 0 && seconds <= 2, "invalid wait duration");
      await tab.playwright.waitForTimeout(seconds * 1000);
      return { waited_seconds: seconds };
    }
    case "shutdown":
      return { shutting_down: true };
    default:
      throw new Error(`unsupported bridge action: ${action}`);
  }
}

export async function startExtensionExecutionBridge({
  tab,
  token,
  host = "127.0.0.1",
  port = 17731,
} = {}) {
  assert(tab?.playwright, "a claimed Chrome tab is required");
  assert(TOKEN_RE.test(String(token || "")), "bridge token must be 64 lowercase hex characters");
  assert(host === "127.0.0.1", "bridge must bind to IPv4 loopback");
  assert(Number.isInteger(port) && port >= 1024 && port <= 65535, "invalid bridge port");

  let resolveDone;
  const done = new Promise((resolve) => {
    resolveDone = resolve;
  });
  let commandChain = Promise.resolve();
  let stopping = false;

  const server = http.createServer((request, response) => {
    commandChain = commandChain.then(async () => {
      const send = (status, body) => {
        response.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
        response.end(JSON.stringify(body));
      };
      try {
        assert(request.method === "POST" && request.url === "/rpc", "unsupported request");
        assert(!stopping, "bridge is shutting down");
        assert(
          safeEqual(request.headers["x-neutralgrid-bridge-token"], token),
          "bridge authentication failed",
        );
        const chunks = [];
        let size = 0;
        for await (const chunk of request) {
          size += chunk.length;
          assert(size <= 65536, "request body is too large");
          chunks.push(chunk);
        }
        const message = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        assert(message?.schema_version === SCHEMA, "bridge schema mismatch");
        assert(typeof message.action === "string", "bridge action is missing");
        const payload = message.payload && typeof message.payload === "object" ? message.payload : {};
        const result = await dispatch(tab, message.action, payload);
        send(200, { schema_version: SCHEMA, ok: true, result });
        if (message.action === "shutdown") {
          stopping = true;
          server.close();
        }
      } catch (error) {
        send(400, {
          schema_version: SCHEMA,
          ok: false,
          error: String(error?.message || error),
        });
      }
    }).catch(() => {
      if (!response.headersSent) response.destroy();
    });
  });
  server.on("close", () => resolveDone());
  server.listen(port, host);
  await new Promise((resolve, reject) => {
    server.once("listening", resolve);
    server.once("error", reject);
  });
  return {
    endpoint: `http://${host}:${port}`,
    done,
    close: () => server.close(),
  };
}
