// =============================================
// Dhaka Building Height Analysis - SPA App
// =============================================

let selectedFile = null;
let lastCounts = null;
let pieChartInstance = null;
let barChartInstance = null;
let reportPieInstance = null;
let radarChartInstance = null;

// ---- SPA Navigation ----
const pages = {
    dashboard: { id: 'page-dashboard', title: 'Dashboard' },
    upload: { id: 'page-upload', title: 'Upload Image' },
    detection: { id: 'page-detection', title: 'Building Detection' },
    analysis: { id: 'page-analysis', title: 'Height Analysis' },
    map: { id: 'page-map', title: 'Dhaka Map' },
    reports: { id: 'page-reports', title: 'Reports' },
    'urban-insights': { id: 'page-urban-insights', title: 'Urban Insights' },
    about: { id: 'page-about', title: 'About Project' }
};

function navigateTo(pageKey) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    // Show target page
    const page = pages[pageKey];
    if (!page) return;
    document.getElementById(page.id)?.classList.add('active');
    document.getElementById('page-title').textContent = page.title;

    // Highlight correct nav item
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.dataset.page === pageKey) item.classList.add('active');
    });

    // Re-init icons after page switch
    lucide.createIcons();
}

// ---- Accordion Toggle (Urban Insights Roadmap) ----
function toggleAccordion(headerBtn) {
    const item = headerBtn.closest('.accordion-item');
    const isOpen = item.classList.contains('open');
    // Close all
    document.querySelectorAll('.accordion-item').forEach(i => i.classList.remove('open'));
    // Toggle current
    if (!isOpen) item.classList.add('open');
    lucide.createIcons();
}

document.addEventListener('DOMContentLoaded', () => {

    // Wire up sidebar nav
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
        item.addEventListener('click', e => {
            e.preventDefault();
            navigateTo(item.dataset.page);
        });
    });

    // ---- Tab toggles for Insight Cards ----
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const card = this.dataset.card;   // 'low' | 'mid' | 'high'
            const tab  = this.dataset.tab;    // 'benefits' | 'disadvantages'

            // Deactivate sibling tabs
            this.closest('.tab-toggles').querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            // Hide both lists for this card, show chosen
            const benefitsList = document.getElementById(`${card}-benefits`);
            const disadvList   = document.getElementById(`${card}-disadvantages`);
            if (benefitsList) benefitsList.classList.remove('active');
            if (disadvList)   disadvList.classList.remove('active');

            const target = tab === 'benefits' ? benefitsList : disadvList;
            if (target) target.classList.add('active');
        });
    });

    // ---- Initialize Radar Chart (static data) ----
    const radarCtx = document.getElementById('radarChart')?.getContext('2d');
    if (radarCtx) {
        radarChartInstance = new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['Land Efficiency', 'Disaster Safety', 'Fire Accessibility', 'Utility Load', 'Livability'],
                datasets: [
                    {
                        label: 'Low-rise',
                        data: [20, 90, 95, 85, 70],
                        backgroundColor: 'rgba(34,197,94,0.18)',
                        borderColor: '#22c55e',
                        pointBackgroundColor: '#22c55e',
                        borderWidth: 2,
                        pointRadius: 4
                    },
                    {
                        label: 'Mid-rise',
                        data: [60, 65, 75, 60, 85],
                        backgroundColor: 'rgba(234,179,8,0.18)',
                        borderColor: '#eab308',
                        pointBackgroundColor: '#eab308',
                        borderWidth: 2,
                        pointRadius: 4
                    },
                    {
                        label: 'High-rise',
                        data: [95, 35, 40, 30, 55],
                        backgroundColor: 'rgba(239,68,68,0.18)',
                        borderColor: '#ef4444',
                        pointBackgroundColor: '#ef4444',
                        borderWidth: 2,
                        pointRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            font: { family: 'Inter', size: 11 },
                            color: '#94a3b8'
                        },
                        pointLabels: {
                            font: { family: 'Inter', size: 12, weight: '600' },
                            color: '#0f172a'
                        },
                        grid: { color: '#e2e8f0' },
                        angleLines: { color: '#e2e8f0' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // ---- Drag & Drop (both drop zones) ----
    ['drop-zone', 'drop-zone-2'].forEach(id => {
        const zone = document.getElementById(id);
        if (!zone) return;
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
            zone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); });
        });
        ['dragenter', 'dragover'].forEach(ev => {
            zone.addEventListener(ev, () => {
                zone.style.borderColor = 'var(--blue-500)';
                zone.style.backgroundColor = 'var(--blue-50)';
            });
        });
        ['dragleave', 'drop'].forEach(ev => {
            zone.addEventListener(ev, () => {
                zone.style.borderColor = '#cbd5e1';
                zone.style.backgroundColor = '#f8fafc';
            });
        });
        zone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    });

    document.getElementById('file-input')?.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (!files.length) return;
        selectedFile = files[0];
        const name = selectedFile.name;

        // Update both upload areas
        const t1 = document.getElementById('upload-status-text');
        const t2 = document.getElementById('upload-status-2');
        if (t1) { t1.textContent = `Selected: ${name}`; t1.style.fontWeight = '600'; t1.style.color = 'var(--text-main)'; }
        if (t2) { t2.textContent = `Selected: ${name}`; t2.style.fontWeight = '600'; t2.style.color = 'var(--text-main)'; }

        // Show preview on upload page
        const preview = document.getElementById('upload-preview');
        const previewCard = document.getElementById('preview-card');
        if (preview && previewCard) {
            preview.src = URL.createObjectURL(selectedFile);
            previewCard.style.display = 'block';
        }

        // Also preview original image on dashboard
        const origImg = document.getElementById('result-orig');
        if (origImg) origImg.src = URL.createObjectURL(selectedFile);
    }

    // ---- Analyze Button (Dashboard + Upload page) ----
    ['analyze-btn', 'analyze-btn-2'].forEach(id => {
        const btn = document.getElementById(id);
        if (!btn) return;
        btn.addEventListener('click', () => runAnalysis());
    });

    async function runAnalysis() {
        if (!selectedFile) {
            alert('Please select an image first.');
            navigateTo('upload');
            return;
        }

        const btns = ['analyze-btn', 'analyze-btn-2'].map(id => document.getElementById(id)).filter(Boolean);
        btns.forEach(b => { b.innerHTML = '<i data-lucide="loader" class="spin"></i> Analyzing...'; b.disabled = true; });
        lucide.createIcons();

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            const response = await fetch('http://localhost:5000/api/analyze', { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            const data = await response.json();
            if (data.error) { alert(`Backend Error: ${data.error}`); return; }

            // Update all result images (dashboard + dedicated pages)
            ['result-orig'].forEach(id => { const el = document.getElementById(id); if (el) el.src = data.original; });
            ['result-detect', 'detect-page-img'].forEach(id => { const el = document.getElementById(id); if (el) el.src = data.detected; });
            ['result-class', 'analysis-page-img'].forEach(id => { const el = document.getElementById(id); if (el) el.src = data.classified; });

            // Per-class images
            const highImg = document.getElementById('high-only-img');
            const medImg  = document.getElementById('medium-only-img');
            const lowImg  = document.getElementById('low-only-img');
            if (highImg && data.high_only)   highImg.src   = data.high_only;
            if (medImg  && data.medium_only) medImg.src    = data.medium_only;
            if (lowImg  && data.low_only)    lowImg.src    = data.low_only;

            // Update metrics
            if (data.counts) {
                lastCounts = data.counts;
                updateMetrics(data.counts);
                renderCharts(data.counts);
                updateReport(data.counts);
                showDetectionResults(data.counts);
                showAnalysisResults();
                updateSuggestionBox(data.counts);
            }

            btns.forEach(b => { b.innerHTML = '<i data-lucide="check"></i> Analysis Complete'; });
            lucide.createIcons();

        } catch (err) {
            console.error(err);
            alert('Could not connect to backend. Make sure "python backend.py" is running on port 5000.');
            btns.forEach(b => { b.innerHTML = '<i data-lucide="alert-triangle"></i> Error'; });
            lucide.createIcons();
        } finally {
            setTimeout(() => {
                btns.forEach(b => { b.innerHTML = '<i data-lucide="cpu"></i> Analyze Image'; b.disabled = false; });
                lucide.createIcons();
            }, 3000);
        }
    }

    function updateMetrics(counts) {
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('m-total', counts.total);
        set('m-low', counts.low);
        set('m-mid', counts.mid);
        set('m-high', counts.high);
    }

    function showDetectionResults(counts) {
        document.getElementById('detection-no-result').style.display = 'none';
        document.getElementById('detection-result').style.display = 'block';
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('d-total', counts.total);
        set('d-low',  counts.low);
        set('d-mid',  counts.mid);
        set('d-high', counts.high);
        // Per-class count badges
        set('panel-high-count', counts.high);
        set('panel-mid-count',  counts.mid);
        set('panel-low-count',  counts.low);
        // Show advisory panel
        updateAdvisoryPanel(counts);
        lucide.createIcons();
    }

    function showAnalysisResults() {
        document.getElementById('analysis-no-result').style.display = 'none';
        document.getElementById('analysis-result').style.display = 'block';
        lucide.createIcons();
    }

    function renderCharts(counts) {
        const labels = ['Low-rise (1-4F)', 'Mid-rise (5-12F)', 'High-rise (13+F)'];
        const values = [counts.low, counts.mid, counts.high];
        const colors = ['#22c55e', '#eab308', '#ef4444'];
        const chartOpts = {
            responsive: true,
            plugins: { legend: { position: 'bottom', labels: { font: { family: 'Inter' } } } }
        };

        // Pie chart (analysis page)
        if (pieChartInstance) pieChartInstance.destroy();
        const pieCtx = document.getElementById('pieChart')?.getContext('2d');
        if (pieCtx) {
            pieChartInstance = new Chart(pieCtx, {
                type: 'doughnut',
                data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }] },
                options: chartOpts
            });
        }

        // Bar chart (analysis page)
        if (barChartInstance) barChartInstance.destroy();
        const barCtx = document.getElementById('barChart')?.getContext('2d');
        if (barCtx) {
            barChartInstance = new Chart(barCtx, {
                type: 'bar',
                data: { labels, datasets: [{ label: 'Buildings', data: values, backgroundColor: colors, borderRadius: 8, borderSkipped: false }] },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }
                }
            });
        }

        // Report pie chart
        if (reportPieInstance) reportPieInstance.destroy();
        const rPieCtx = document.getElementById('reportPieChart')?.getContext('2d');
        if (rPieCtx) {
            reportPieInstance = new Chart(rPieCtx, {
                type: 'doughnut',
                data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }] },
                options: chartOpts
            });
        }
    }

    function updateReport(counts) {
        const total = counts.total || 1;
        const rows = [
            { label: 'Low-rise (1–4 Floors)', count: counts.low, color: '#22c55e' },
            { label: 'Mid-rise (5–12 Floors)', count: counts.mid, color: '#eab308' },
            { label: 'High-rise (13+ Floors)', count: counts.high, color: '#ef4444' },
            { label: 'Total', count: total, color: 'var(--blue-500)' }
        ];
        const tbody = document.getElementById('report-table-body');
        if (tbody) {
            tbody.innerHTML = rows.map(r => `
                <tr>
                    <td><span style="display:inline-block;width:10px;height:10px;background:${r.color};border-radius:2px;margin-right:8px;vertical-align:middle;"></span>${r.label}</td>
                    <td><strong>${r.count}</strong></td>
                    <td>${r.label === 'Total' ? '100%' : ((r.count / total) * 100).toFixed(1) + '%'}</td>
                </tr>
            `).join('');
        }
        const placeholder = document.getElementById('report-placeholder');
        const content = document.getElementById('report-content');
        if (placeholder) placeholder.style.display = 'none';
        if (content) content.style.display = 'block';

        // Populate policy brief
        updatePolicyBrief(counts);
    }

    // ---- Image Modal ----
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    let currentScale = 1;

    function setZoom(scale) {
        currentScale = Math.min(Math.max(scale, 0.3), 6);
        modalImg.style.transform = `scale(${currentScale})`;
        const lbl = document.getElementById('zoom-level');
        if (lbl) lbl.textContent = Math.round(currentScale * 100) + '%';
    }

    function openModal(src) {
        if (!src || src.endsWith('original.jpg') || src.endsWith('detected.jpg') || src.endsWith('classification.jpg')) {
            alert('Please run an analysis first to view this image.');
            return;
        }
        modal.style.display = 'flex';
        modalImg.src = src;
        setZoom(1);
    }

    function closeModal() { modal.style.display = 'none'; setZoom(1); }

    document.querySelectorAll('.result-image-wrapper').forEach(w => {
        w.style.cursor = 'zoom-in';
        w.addEventListener('click', function () {
            const img = this.querySelector('img');
            if (img?.src) openModal(img.src);
        });
    });

    document.getElementById('close-modal-btn')?.addEventListener('click', closeModal);
    modal?.addEventListener('click', e => { if (e.target === modal) closeModal(); });
    document.getElementById('zoom-in-btn')?.addEventListener('click', e => { e.stopPropagation(); setZoom(currentScale + 0.25); });
    document.getElementById('zoom-out-btn')?.addEventListener('click', e => { e.stopPropagation(); setZoom(currentScale - 0.25); });
    document.getElementById('zoom-reset-btn')?.addEventListener('click', e => { e.stopPropagation(); setZoom(1); });
    modal?.addEventListener('wheel', e => {
        if (modal.style.display === 'flex') { e.preventDefault(); setZoom(currentScale + (e.deltaY > 0 ? -0.15 : 0.15)); }
    }, { passive: false });
    document.addEventListener('keydown', e => {
        if (modal.style.display !== 'flex') return;
        if (e.key === 'Escape') closeModal();
        if (e.key === '+' || e.key === '=') setZoom(currentScale + 0.25);
        if (e.key === '-') setZoom(currentScale - 0.25);
    });

    // Initialize icons on load
    lucide.createIcons();
});

// =============================================
// FEATURE 2: Smart City Advisory Panel
// =============================================

function updateAdvisoryPanel(counts) {
    const panel = document.getElementById('advisory-panel');
    if (!panel) return;

    const total = counts.total || 1;
    const lowPct  = (counts.low  / total) * 100;
    const midPct  = (counts.mid  / total) * 100;
    const highPct = (counts.high / total) * 100;

    // Determine zone profile
    let zoneLabel, zoneClass, riskClass, recommendations;

    if (lowPct > 60) {
        // Low-rise dominant
        zoneLabel = 'Low-Rise Dominant Zone';
        zoneClass = 'badge-green';
        riskClass = 'advisory-green';
        recommendations = [
            'Land inefficiency detected — over 60% low-rise stock wastes scarce urban land. Initiate DAP-aligned TOD rezoning for mid-rise transition corridors.',
            'Plan Transit-Oriented mid-rise clusters (5–8 floors) along main arterials to gradually increase density without overburdening infrastructure.',
            'Introduce green buffer zones and community parks freed from horizontal sprawl to improve livability and flood resilience.'
        ];
    } else if (highPct > 40) {
        // High-rise dominant
        zoneLabel = 'High-Rise Cluster Zone';
        zoneClass = 'badge-red';
        riskClass = 'advisory-red';
        recommendations = [
            'CRITICAL: High-rise concentration exceeds 40%. Mandate liquefaction micro-zonation audits and BNBC-2020 structural compliance reviews immediately.',
            'Restrict further high-rise permits to within 500m radius of MRT stations only — enforce TOD zoning buffers strictly.',
            'Audit fire emergency access routes — ensure all high-rise clusters have minimum 20ft road clearance for Fire Service skylift deployment.'
        ];
    } else {
        // Mixed / balanced
        zoneLabel = 'Mixed Urban Zone';
        zoneClass = 'badge-yellow';
        riskClass = 'advisory-yellow';
        recommendations = [
            'Balanced zone detected — maintain diversity with DAP-compliant setbacks and road-width planning.',
            'Leverage mid-rise density for social cohesion; prioritize pedestrian infrastructure and neighborhood-scale amenities.',
            'Promote green roof mandates and rainwater harvesting across mid/high-rise buildings to meet Sponge City targets.'
        ];
    }

    // Remove previous risk class
    panel.classList.remove('advisory-green', 'advisory-yellow', 'advisory-red');
    panel.classList.add(riskClass);

    // Zone badge
    const badge = document.getElementById('advisory-zone-badge');
    if (badge) {
        badge.textContent = zoneLabel;
        badge.className = `zone-badge ${zoneClass}`;
    }

    // Stats
    const statsEl = document.getElementById('advisory-stats');
    if (statsEl) {
        statsEl.innerHTML = `
            <strong>${counts.total}</strong> buildings detected &nbsp;·&nbsp;
            <span style="color:#22c55e;">Low-rise: ${lowPct.toFixed(1)}%</span> &nbsp;|&nbsp;
            <span style="color:#ca8a04;">Mid-rise: ${midPct.toFixed(1)}%</span> &nbsp;|&nbsp;
            <span style="color:#dc2626;">High-rise: ${highPct.toFixed(1)}%</span>
        `;
    }

    // Recommendations
    const recsEl = document.getElementById('advisory-recs');
    if (recsEl) {
        recsEl.innerHTML = recommendations.map(r => `<li>${r}</li>`).join('');
    }

    panel.style.display = 'block';
    lucide.createIcons();
}

// =============================================
// FEATURE 3: Smart City Policy Brief
// =============================================

function updatePolicyBrief(counts) {
    const brief = document.getElementById('policy-brief');
    const briefBody = document.getElementById('policy-brief-body');
    if (!brief || !briefBody) return;

    const total = counts.total || 1;
    const lowPct  = ((counts.low  / total) * 100).toFixed(1);
    const midPct  = ((counts.mid  / total) * 100).toFixed(1);
    const highPct = ((counts.high / total) * 100).toFixed(1);

    const sections = [
        {
            label: 'Low-rise (1–4 Floors)',
            headerClass: 'ps-green',
            count: counts.low,
            pct: lowPct,
            recs: [
                'Enforce DAP 2022–2035 low-rise preservation zones near river buffer areas (Buriganga, Turag, Balu) to prevent flood-plain encroachment.',
                'Introduce incentivized voluntary amalgamation programs for low-rise landowners to consolidate plots for mid-rise development.',
                'Upgrade drainage and sanitation infrastructure in low-rise dense clusters (Demra, Jatrabari, Mirpur) to reduce per-capita utility cost.'
            ]
        },
        {
            label: 'Mid-rise (4–8 Floors)',
            headerClass: 'ps-yellow',
            count: counts.mid,
            pct: midPct,
            recs: [
                'Mandate minimum 12ft setbacks on all mid-rise developments on roads below 30ft to preserve emergency vehicle access.',
                'Introduce mandatory structural audit every 10 years for mid-rise buildings in high-density zones (Mohammadpur, Rayer Bazar, Malibagh).',
                'Encourage mixed-use ground floors (retail + residential) in mid-rise clusters to improve walkability and neighborhood economic vitality.'
            ]
        },
        {
            label: 'High-rise (9+ Floors)',
            headerClass: 'ps-red',
            count: counts.high,
            pct: highPct,
            recs: [
                'Strictly limit new high-rise permits to RAJUK-designated zones with road widths ≥ 40ft and confirmed MRT proximity within 500m.',
                'Mandate automatic fire suppression systems, seismic P-wave auto-shutoff valves, and helicopter landing pads for all high-rises above 15 floors.',
                'Enforce WASA utility capacity assessments before approval — no high-rise permit without certified water/sewage upgrade commitment.'
            ]
        }
    ];

    briefBody.innerHTML = sections.map(s => `
        <div class="policy-section">
            <div class="policy-section-header ${s.headerClass}">
                <span>${s.label}</span>
            </div>
            <div class="policy-section-body">
                <div class="policy-stats-row">
                    <div class="policy-stat">
                        <span class="policy-stat-label">Count</span>
                        <span class="policy-stat-value">${s.count}</span>
                    </div>
                    <div class="policy-stat">
                        <span class="policy-stat-label">Share</span>
                        <span class="policy-stat-value">${s.pct}%</span>
                    </div>
                    <div class="policy-stat">
                        <span class="policy-stat-label">Total Buildings</span>
                        <span class="policy-stat-value">${total}</span>
                    </div>
                </div>
                <div class="policy-recs-label">Policy Recommendations</div>
                <ul class="policy-recs-list">
                    ${s.recs.map(r => `<li>${r}</li>`).join('')}
                </ul>
            </div>
        </div>
    `).join('');

    brief.style.display = 'block';
    lucide.createIcons();
}

// =============================================
// Dynamic Suggestion Box (Based on Uploaded Image)
// =============================================

function updateSuggestionBox(counts) {
    if (!counts) return;
    const banner = document.getElementById('sug-result-banner');
    const subtitle = document.getElementById('sug-subtitle');
    const total = counts.total || 0;

    if (total === 0) {
        if (banner) banner.style.display = 'none';
        return;
    }

    const low = counts.low || 0;
    const mid = counts.mid || 0;
    const high = counts.high || 0;

    const lowPct = ((low / total) * 100).toFixed(1);
    const midPct = ((mid / total) * 100).toFixed(1);
    const highPct = ((high / total) * 100).toFixed(1);

    // Update counts inside the banner
    const elH = document.getElementById('srb-high');
    const elM = document.getElementById('srb-mid');
    const elL = document.getElementById('srb-low');
    const elT = document.getElementById('srb-total');
    if (elH) elH.textContent = `${high} (${highPct}%)`;
    if (elM) elM.textContent = `${mid} (${midPct}%)`;
    if (elL) elL.textContent = `${low} (${lowPct}%)`;
    if (elT) elT.textContent = total;

    // Update tab badges with detected counts from this image
    const tcH = document.getElementById('tab-cnt-high');
    const tcM = document.getElementById('tab-cnt-med');
    const tcL = document.getElementById('tab-cnt-low');
    if (tcH) { tcH.textContent = high; tcH.style.display = 'inline-block'; }
    if (tcM) { tcM.textContent = mid;  tcM.style.display = 'inline-block'; }
    if (tcL) { tcL.textContent = low;  tcL.style.display = 'inline-block'; }

    // Determine dominant type and contextual recommendation based on this uploaded image
    let dominantName = 'Low-rise';
    let dominantTarget = 'sug-low';
    let recText = '';

    if (high >= mid && high >= low && high > 0) {
        dominantName = 'High-rise';
        dominantTarget = 'sug-high';
        recText = `Detected from your uploaded image: <strong>High-Rise Dominant Cluster (${high} of ${total} buildings, ${highPct}%)</strong>. Key Dhaka city recommendation: High density requires strict BNBC-2020 seismic liquefaction compliance and emergency fire ladder clearance (>20 ft road access). Prioritize vertical expansion only around MRT transit corridors.`;
    } else if (mid >= low && mid >= high && mid > 0) {
        dominantName = 'Mid-rise';
        dominantTarget = 'sug-med';
        recText = `Detected from your uploaded image: <strong>Mid-Rise Dominant Zone (${mid} of ${total} buildings, ${midPct}%)</strong>. Key Dhaka city recommendation: Optimal density balance for Dhaka's secondary roads. Ensure minimum 25–30 ft street widths to prevent parking bottlenecks and maintain natural cross-ventilation.`;
    } else {
        dominantName = 'Low-rise';
        dominantTarget = 'sug-low';
        recText = `Detected from your uploaded image: <strong>Low-Rise Dominant Zone (${low} of ${total} buildings, ${lowPct}%)</strong>. Key Dhaka city recommendation: High structural seismic safety, but low land efficiency. If located in prime urban corridors, consider phased DAP-guided transition to mid-rise TOD clusters to curb horizontal encroachment on wetlands.`;
    }

    const titleEl = document.getElementById('srb-title');
    if (titleEl) titleEl.textContent = `Analysis of Uploaded Image: ${dominantName} Dominant Zone (${total} Buildings Detected)`;

    const recEl = document.getElementById('srb-recommendation');
    if (recEl) recEl.innerHTML = recText;

    if (subtitle) {
        subtitle.innerHTML = `Insights customized for your uploaded image — <strong>${dominantName}</strong> is dominant (${total} total detected buildings).`;
    }

    // Auto-switch tab to the dominant building type detected
    const targetTabBtn = document.querySelector(`.stab[data-target="${dominantTarget}"]`);
    if (targetTabBtn) {
        switchSugTab(targetTabBtn, dominantTarget);
    }

    if (banner) {
        banner.style.display = 'block';
    }
    lucide.createIcons();
}

// ---- Global: Suggestion Box Tab Switcher ----
function switchSugTab(btn, targetId) {
    // Deactivate all tabs
    document.querySelectorAll('.stab').forEach(t => t.classList.remove('active'));
    // Deactivate all panels
    document.querySelectorAll('.sug-panel').forEach(p => p.classList.remove('active'));
    // Activate clicked tab and matching panel
    btn.classList.add('active');
    const panel = document.getElementById(targetId);
    if (panel) panel.classList.add('active');
}

// ---- Global: Download Image ----
function downloadImage(imgId, filename) {
    const img = document.getElementById(imgId);
    if (!img?.src || img.src.endsWith('original.jpg') || img.src.endsWith('detected.jpg') || img.src.endsWith('classification.jpg')) {
        alert('Please run an analysis first to download this image.');
        return;
    }
    const a = document.createElement('a');
    a.href = img.src;
    a.download = filename;
    a.click();
}

// ---- Global: Download CSV Report ----
function downloadReport() {
    const rows = document.querySelectorAll('#report-table-body tr');
    if (!rows.length) return;
    let csv = 'Category,Count,Percentage\n';
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        csv += Array.from(cells).map(c => c.textContent.replace(/\s+/g, ' ').trim()).join(',') + '\n';
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'dhaka_building_report.csv';
    a.click();
}
