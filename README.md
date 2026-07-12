# SeisMind

**The agent-native seismic interpretation workstation for Windows.**

SeisMind is a desktop application for seismic interpretation and reservoir evaluation — load industry-standard data, tie wells, pick horizons, interpret faults, and build complete interpretation projects. It's **agent-native**: you can drive it in plain English through an AI coding agent (like Claude Code) that talks to SeisMind over the Model Context Protocol. Everything runs locally — your data never leaves your machine.

## Download

**[⬇ SeisMind Free 3.1.0 — Windows 64-bit installer](https://github.com/narunbabu/geoagent/releases/download/seismind-v3.1.0/SeisMind-Free-Setup-3.1.0.exe)** (~180 MB)

A single installer — no Python, no dependencies. Runs on Windows 10/11 (64-bit).

> The installer is currently unsigned, so Windows SmartScreen may show an "unknown publisher" warning. Choose **More info → Run anyway** to install.

---

## Work with SeisMind through an AI agent

SeisMind ships an MCP server, so an AI coding agent can operate it for you. Point your agent (e.g. Claude Code) at SeisMind once:

```jsonc
// add SeisMind as an MCP server (e.g. .mcp.json / your agent's config)
{
  "mcpServers": {
    "seismind": { "command": "seismind", "args": ["serve"] }
  }
}
```

Now you can just describe what you want. Two prompts take you from a clean install to a real, interpreted project — both work in the **Free** edition, on a public dataset, so you can try the whole flow before using your own data.

### 1 · Get started — download real data and build a project

> **Prompt:** *"Download the Teapot Dome sample dataset, build a SeisMind project from it, and open it."*

Your agent calls `download_sample_data` → `build_sample_project` → `open_project_in_gui`. SeisMind fetches the public **Teapot Dome (NPR-3)** field data — a US DOE / RMOTC public-domain 3D survey with well logs — builds a complete SeisMind project (seismic volume, wells, logs, wavelets), and opens it in the desktop app. First run downloads ~1.7 GB; it's cached after that.

### 2 · Preliminary interpretation

> **Prompt:** *"Run a preliminary interpretation on the Teapot Dome project — give me a well-location map, a representative seismic section, and a summary of the survey."*

Your agent calls `preliminary_interpretation`, which writes a **well-location basemap**, a **representative seismic section** (an inline near a well), and a **summary** (survey extent, inline/crossline range, sample rate, well and log inventory) into the project's `interpretation/` folder — a quick-look you can build on.

### Going further — SeisMind Professional

The same conversational workflow scales into the paid tier:

> **Prompt (Pro):** *"Predict a porosity volume from the wells with machine learning, extract the key seismic attributes, correlate the target interval across all wells, and build a client-ready PowerPoint of the results."*

Pro adds autonomous multi-step agents, ML log/volume prediction, seismic attributes, petrophysics, unlimited data, and presentation/report generation. → [ameyem.com/products/seismind](https://ameyem.com/products/seismind)

---

## Editions

| | Free | Professional | Enterprise |
|---|:---:|:---:|:---:|
| Seismic & well data (SEG-Y, LAS, ZMAP, Petrel ASCII) | ✔ | ✔ | ✔ |
| Horizon picking, fault interpretation, well tie | ✔ | ✔ | ✔ |
| **Agent: download sample data, build project, preliminary interpretation** | ✔ | ✔ | ✔ |
| Project size | 5 wells, 1 seismic volume | Unlimited | Unlimited |
| ML log & volume prediction | — | ✔ | ✔ |
| Seismic attributes & petrophysics | — | ✔ | ✔ |
| Autonomous multi-step agents | — | ✔ | ✔ |
| Presentation & report generation | — | ✔ | ✔ |
| On-prem deployment, SSO, custom model training | — | — | ✔ |

**SeisMind Professional & Enterprise** → [ameyem.com/products/seismind](https://ameyem.com/products/seismind)

## Data formats

Import/export industry standards: **SEG-Y** (seismic), **LAS** (well logs), **ZMAP** (horizon grids), and **Petrel ASCII** (well heads, deviation, checkshots, tops).

## Privacy

Your seismic and well data stay on your machine. SeisMind reads and writes local project files; nothing is uploaded.

## Support & feedback

Bug reports and feature requests welcome — [open an issue](https://github.com/narunbabu/geoagent/issues).

---

© 2026 Ameyem Geosolutions. All rights reserved. SeisMind is proprietary software; the Free edition is distributed free of charge under the license terms bundled with the installer.
