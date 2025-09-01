const { createApp, ref, reactive, watch, computed, onMounted, nextTick } = Vue;

const debounce = (func, delay) => {
    let timeoutId;
    return (...args) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, delay);
    };
};

const app = createApp({
    setup() {
        let worker;
        let mainChart = null;
        let loadingTimer = null;

        const chartContainer = ref(null);
        const dashboardGridRef = ref(null);
        const dropdownContainer = ref(null);

        const dataState = reactive({
            file: null, dataUrl: '', records: [], dimensions: [], measures: [], calculatedFields: [],
        });
        const uiState = reactive({
            isLoading: false, viewMode: 'worksheet', isDirty: false,
            contextMenu: { visible: false, top: 0, left: 0, field: null },
            isDropdownVisible: false, editingId: null, editText: '',
            tableData: { headers: [], rows: [] },
        });
        const modals = reactive({
            settings: false, grouping: false, calculatedField: false,
            statisticalAnalysis: false,
            binning: false,
        });
        const settings = reactive({
            theme: localStorage.getItem('datacanvas-theme') || 'light',
            activePalette: localStorage.getItem('datacanvas-palette') || 'default',
            activeNumberFormat: localStorage.getItem('datacanvas-format') || 'default',
            showDataLabels: JSON.parse(localStorage.getItem('datacanvas-labels') || 'false'),
            autoUpdateEnabled: JSON.parse(localStorage.getItem('datacanvas-autoupdate') || 'true'),
        });

        const worksheets = ref([]);
        const activeWorksheetId = ref(null);
        let worksheetCounter = 1;
        const dashboardLayout = ref([]);
        const dashboardCharts = new Map();
        const dashboardFilters = reactive({});

        const activeFilter = ref(null);
        const newCalcField = reactive({ name: '', formula: '' });
        const groupingState = reactive({ field: '', uniqueValues: [], selectedValues: [], newGroupName: '' });
        const gridInteraction = reactive({
            active: false, type: null, item: null, startX: 0, startY: 0, initialX: 0, initialY: 0,
            initialW: 0, initialH: 0, gridCellWidth: 0, gridCellHeight: 0, animationFrameId: null,
        });

        const analysisState = reactive({
            testType: 'correlation',
            measure1: '',
            measure2: '',
            ttest: { measure: '', dimension: '' },
            zscore: { measure: '' },
            clustering: { k: 3, fields: [] },
            result: null,
            error: '',
        });

        const binningState = reactive({
            measure: '',
            binSize: null,
            binName: ''
        });

        const isTimeTraveling = false;
        const history = reactive([]);
        const historyIndex = ref(-1);

        const palettes = {
            default: ['#40a9ff', '#1890ff', '#096dd9', '#0050b3', '#003a8c', '#13c2c2', '#08979c', '#006d75', '#00474f'],
            sunset: ['#f94144', '#f3722c', '#f8961e', '#f9c74f', '#43aa8b', '#577590', '#277da1'],
            forest: ['#1b4332', '#2d6a4f', '#40916c', '#52b788', '74c69d', '#95d5b2', '#b7e4c7'],
            colorblindFriendly: ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']
        };
        const numberFormats = {
            default: 'Default (1,234.5)', usd: 'USD ($1,234.50)', eur: 'EUR (€1,234.50)',
            gbp: 'GBP (£1,234.50)', jpy: 'JPY (¥1,234)', percent: 'Percent (12.3%)'
        };
        const dateLevels = ['year', 'quarter', 'month'];

        const activeWorksheet = computed(() => worksheets.value.find(w => w.id === activeWorksheetId.value));
        const activeShelves = computed(() => activeWorksheet.value?.shelves);
        const activeChartType = computed({
            get: () => activeWorksheet.value?.activeChartType,
            set: (val) => { if (activeWorksheet.value) activeWorksheet.value.activeChartType = val; }
        });
        const chartConfigured = computed(() => activeShelves.value && (activeShelves.value.columns.length > 0 || activeShelves.value.rows.length > 0));
        const isDarkTheme = computed({
            get: () => settings.theme === 'dark',
            set: (newValue) => { settings.theme = newValue ? 'dark' : 'light'; }
        });
        const canUndo = computed(() => historyIndex.value > 0);
        const canRedo = computed(() => historyIndex.value < history.length - 1);
        const isAnalyticsPanelVisible = computed(() => activeWorksheet.value?.activeChartType === 'line' && activeShelves.value?.columns.some(f => f.isDate));
        const chartSuggestions = computed(() => {
            if (!activeShelves.value) return [];
            const suggestions = [];
            const allFields = [...activeShelves.value.columns, ...activeShelves.value.rows, ...activeShelves.value.color];
            const dims = allFields.filter(f => f.type === 'dimension').length;
            const measuresCount = allFields.filter(f => f.type === 'measure').length;
            if (dims >= 1 && measuresCount >= 1) {
                suggestions.push({ name: 'Table', type: 'table', icon: '📇' }, { name: 'Bar Chart', type: 'bar', icon: '📊' }, { name: 'Line Chart', type: 'line', icon: '📈' }, { name: 'Area Chart', type: 'area', icon: '📉' }, { name: 'Treemap', type: 'treemap', icon: '🟫' }, { name: 'Box Plot', type: 'boxplot', icon: ' 箱' });
            }
            if (allFields.find(f => f.type === 'dimension' && isGeoField(f.name)) && measuresCount >= 1) {
                suggestions.push({ name: 'Map Chart', type: 'map', icon: '🗺️' });
            }
            if (measuresCount >= 2) {
                if (activeShelves.value.rows.filter(f => f.type === 'measure').length === 2 && dims >= 1) {
                    suggestions.push({ name: 'Combo Chart', type: 'combo', icon: '⬱' });
                }
                suggestions.push({ name: 'Scatter Plot', type: 'scatter', icon: '✨' });
            }
                    if (dims >= 2 && measuresCount >= 1) {
                suggestions.push({ name: 'Sankey Diagram', type: 'sankey', icon: '🌊' });
            }
            if (dims >= 1 && allFields.filter(f => isDateField(f.name)).length >= 2) {
                suggestions.push({ name: 'Gantt Chart', type: 'gantt', icon: '📊' });
            }
            if (dims >= 1 && measuresCount === 0) {
                suggestions.push({ name: 'Word Cloud', type: 'wordCloud', icon: '☁️' });
            }
            if (dims >= 2 && measuresCount >= 1) {
                suggestions.push({ name: 'Heatmap', type: 'heatmap', icon: '🔥' });
            }
            if (dims >= 1 && measuresCount >= 1 && !suggestions.some(s => s.type === 'pie')) {
                suggestions.push({ name: 'Pie Chart', type: 'pie', icon: '🥧' });
            }
            return suggestions;
        });
        
        const canRunAnalysis = computed(() => {
            switch (analysisState.testType) {
                case 'correlation':
                    return analysisState.measure1 && analysisState.measure2 && analysisState.measure1 !== analysisState.measure2;
                case 'ttest':
                    return analysisState.ttest.measure && analysisState.ttest.dimension;
                case 'zscore':
                    return analysisState.zscore.measure;
                case 'clustering': 
                    return analysisState.clustering.k > 1 && analysisState.clustering.fields.length >= 2;
                default:
                    return false;
                    
            }
        });

        const setLoading = (isLoading) => {
            clearTimeout(loadingTimer);
            if (isLoading) {
                loadingTimer = setTimeout(() => { uiState.isLoading = true; }, 300);
            } else {
                uiState.isLoading = false;
            }
        };
        const getWorksheetById = (id) => worksheets.value.find(w => w.id === id);
        const isDateField = (fieldName) => fieldName.toLowerCase().includes('date');
        const isGeoField = (fieldName) => ['country', 'nation', 'region', 'state', 'province', 'location'].some(k => fieldName.toLowerCase().includes(k));
        const getSortClass = (field) => field.sort ? `sort-${field.sort}` : '';
        const formatNumber = (value) => {
            if (typeof value !== 'number' || isNaN(value)) return value;
            switch (settings.activeNumberFormat) {
                case 'usd': return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
                case 'eur': return new Intl.NumberFormat('de-DE', { style: 'currency', currency: 'EUR' }).format(value);
                case 'gbp': return new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP' }).format(value);
                case 'jpy': return new Intl.NumberFormat('ja-JP', { style: 'currency', currency: 'JPY' }).format(value);
                case 'percent': return new Intl.NumberFormat('default', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(value);
                default: return new Intl.NumberFormat('default', { minimumFractionDigits: 0, maximumFractionDigits: 1 }).format(value);
            }
        };
        const getTableCellStyle = (value, header) => {
            if (typeof value !== 'number' || !dataState.measures.find(m => m.name === header)) return {};
            if (value < 0) return { color: '#ee6666' };
            if (value > 0) return { color: '#3ba272' };
            return {};
        };
        const getMapCountryName = (name) => ({ "United States": "United States of America", "England": "United Kingdom", "Russia": "Russian Federation", "South Korea": "Korea", "S. Korea": "Korea", "Vietnam": "Viet Nam" }[name] || name);

        const processDataWithWorker = (config) => {
            return new Promise((resolve, reject) => {
                if (!worker) return reject(new Error("Worker is not available."));
                worker.onmessage = (event) => resolve(event.data);
                worker.onerror = (error) => reject(error);
                worker.postMessage({
                    type: 'processDataForChart',
                    payload: {
                        records: JSON.parse(JSON.stringify(dataState.records)),
                        config: JSON.parse(JSON.stringify(config)),
                        measures: JSON.parse(JSON.stringify(dataState.measures)),
                        dashboardFilters: JSON.parse(JSON.stringify(dashboardFilters))
                    }
                });
            });
        };
        const updateVisualization = async () => {
            if (!activeWorksheet.value || !chartConfigured.value) {
                if (mainChart) mainChart.clear();
                uiState.tableData = { headers: [], rows: [] };
                return;
            }
            setLoading(true);
            await nextTick();
            try {
                const { chartData: processedData, payload } = await processDataWithWorker(activeWorksheet.value);
                activeWorksheet.value.chartData = processedData;
                if (activeWorksheet.value.activeChartType === 'table') {
                    if (mainChart) mainChart.clear();
                    uiState.tableData = processedData.length > 0 ? { headers: Object.keys(processedData[0]), rows: processedData } : { headers: [], rows: [] };
                } else {
                    uiState.tableData = { headers: [], rows: [] };
                    if (!chartContainer.value) return;
                    const option = generateChartOption(processedData, payload);
                    if (mainChart) mainChart.dispose();
                    mainChart = echarts.init(chartContainer.value, settings.theme);
                    mainChart.setOption(option, true);
                }
            } catch (e) { console.error("Visualization update failed:", e); } finally { setLoading(false); }
        };
        const debouncedUpdateVisualization = debounce(updateVisualization, 100);
        const renderDashboardCharts = async () => {
            setLoading(true);
            await nextTick();
            try {
                for (const item of dashboardLayout.value) {
                    const worksheet = getWorksheetById(item.worksheetId);
                    if (!worksheet) continue;
                    const { chartData, payload } = await processDataWithWorker(worksheet);
                    worksheet.chartData = chartData;
                    const chartDom = document.getElementById(`chart-container-${item.worksheetId}`);
                    if (chartDom) {
                        const option = generateChartOption(chartData, payload);
                        let dashChart = dashboardCharts.get(item.worksheetId);
                        if (dashChart) dashChart.dispose();
                        dashChart = echarts.init(chartDom, settings.theme);
                        dashChart.setOption(option, true);
                        dashChart.on('click', (params) => handleDashboardChartClick(params, worksheet));
                        dashboardCharts.set(item.worksheetId, dashChart);
                    }
                }
            } catch (e) { console.error("Dashboard render failed:", e); } finally { setLoading(false); }
        };
        const requestVisualizationUpdate = () => {
            if (settings.autoUpdateEnabled) debouncedUpdateVisualization();
            else uiState.isDirty = true;
        };
        const applyManualUpdate = () => { if (uiState.isDirty) { updateVisualization(); uiState.isDirty = false; } };
        const handleShelfUpdate = () => {
            if (!activeWorksheet.value) return;
            const allShelfFields = [...activeShelves.value.columns, ...activeShelves.value.rows];
            allShelfFields.forEach(field => { if (isDateField(field.name) && !field.isDate) { field.isDate = true; field.drillLevel = 'year'; } });
            activeWorksheet.value.activeChartType = (() => {
                const measureCount = allShelfFields.filter(f => f.type === 'measure').length;
                if (measureCount >= 2) return 'scatter';
                if (allShelfFields.some(f => f.isDate) && measureCount >= 1) return 'line';
                if (allShelfFields.filter(f => f.type === 'dimension').length >= 1 && measureCount >= 1) return 'bar';
                return activeWorksheet.value.activeChartType;
            })();
            requestVisualizationUpdate();
        };

        const generateChartOption = (chartData, payload) => {
            const chartType = payload.chart_type;
            const sortField = [...(activeWorksheet.value?.shelves.columns || []), ...(activeWorksheet.value?.shelves.rows || [])].find(f => f.sort);
            if (sortField && Array.isArray(chartData)) {
                chartData.sort((a, b) => {
                    const valA = a[sortField.name] || 0;
                    const valB = b[sortField.name] || 0;
                    const order = sortField.sort === 'asc' ? 1 : -1;
                    return sortField.type === 'measure' ? (valA - valB) * order : String(valA).localeCompare(String(valB)) * order;
                });
            } else if (Array.isArray(chartData) && payload.x_axis && chartType !== 'scatter') {
                chartData.sort((a, b) => String(a[payload.x_axis]).localeCompare(String(b[payload.x_axis])));
            }
            const chartGenerators = { bar: _getBarLineAreaOption, line: _getBarLineAreaOption, area: _getBarLineAreaOption, pie: _getPieOption, scatter: _getScatterOption, heatmap: _getHeatmapOption, treemap: _getTreemapOption, combo: _getComboOption, map: _getMapOption, boxplot: _getBoxPlotOption, sankey: _getSankeyOption, wordCloud: _getWordCloudOption, gantt: _getGanttOption };
            return (chartGenerators[chartType] || (() => ({})))(chartData, payload);
        };
        const _createTooltipFormatter = (params) => `${params[0].axisValueLabel}<br/>` + params.map(p => `${p.marker} ${p.seriesName}: ${formatNumber(p.value)}`).join('<br/>');
        const _getBarLineAreaOption = (chartData, payload) => {
            const { chart_type, x_axis, y_axes, color, analytics } = payload;
            const categories = [...new Set(chartData.map(d => d[x_axis]))];
            const legendData = color ? [...new Set(chartData.map(d => d[color]))] : [];
        
            // Base series for the historical data
            const series = legendData.length > 0
                ? legendData.map(c => ({ 
                    name: c, 
                    type: chart_type === 'area' ? 'line' : chart_type, 
                    stack: 'total', 
                    areaStyle: chart_type === 'area' ? {} : null, 
                    emphasis: { focus: 'series' }, 
                    label: { show: settings.showDataLabels, position: 'top', formatter: p => formatNumber(p.value) }, 
                    data: categories.map(cat => chartData.find(d => d[x_axis] === cat && d[color] === c)?.[y_axes[0]] || 0) 
                }))
                : [{ 
                    name: y_axes[0], 
                    type: chart_type === 'area' ? 'line' : chart_type, 
                    areaStyle: chart_type === 'area' ? {} : null, 
                    emphasis: { focus: 'series' }, 
                    label: { show: settings.showDataLabels, position: 'top', formatter: p => formatNumber(p.value) }, 
                    data: categories.map(cat => chartData.find(d => d[x_axis] === cat)?.[y_axes[0]] || 0) 
                }];
            
            const option = { 
                color: palettes[settings.activePalette], 
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: _createTooltipFormatter }, 
                legend: { data: legendData }, 
                xAxis: { type: 'category', data: categories }, 
                yAxis: { type: 'value', axisLabel: { formatter: formatNumber } }, 
                series, 
                grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true } 
            };
        
            // New: Add forecast series with confidence interval
            if (chart_type === 'line' && analytics?.showTrendLine && analytics?.forecastData && analytics?.forecastConfidence) {
                const forecastCategories = [...categories];
                for(let i = 0; i < analytics.forecastData.length; i++) {
                    forecastCategories.push(`Forecast ${i + 1}`);
                }
                option.xAxis.data = forecastCategories;
        
                // Series for the forecast line
                const forecastLineData = Array(categories.length).fill(null).concat(analytics.forecastData.map(d => d.prediction));
                option.series.push({
                    name: 'Forecast',
                    type: 'line',
                    smooth: true,
                    symbol: 'none',
                    lineStyle: { type: 'dashed', width: 2 },
                    data: forecastLineData
                });
                
                // Series for the confidence interval (shaded area)
                const confidenceIntervalData = Array(categories.length).fill([null, null]).concat(
                    analytics.forecastConfidence.map(ci => [ci.lower, ci.upper])
                );
                option.series.push({
                    name: 'Confidence Interval',
                    type: 'line',
                    smooth: true,
                    symbol: 'none',
                    lineStyle: { opacity: 0 },
                    areaStyle: {
                        color: 'rgba(59, 130, 246, 0.2)' // A light blue shade
                    },
                    data: confidenceIntervalData,
                    markLine: {
                        data: [
                            { yAxis: 'min', name: 'Min', label: { show: false } },
                            { yAxis: 'max', name: 'Max', label: { show: false } }
                        ]
                    }
                });
                
                option.legend.data.push('Forecast');
            } else if (chart_type === 'line' && analytics?.showTrendLine && chartData.length > 1 && !color) {
                const regressionData = chartData.map((d, i) => [i, d[y_axes[0]]]);
                const trendLineFunc = ss.linearRegressionLine(ss.linearRegression(regressionData));
                const trendLineData = regressionData.map(d => trendLineFunc(d[0]));
                option.series.push({ name: 'Trend Line', type: 'line', smooth: true, symbol: 'none', lineStyle: { type: 'dashed', width: 2 }, data: trendLineData });
                option.legend.data.push('Trend Line');
            }
            
            return option;
        };
        const _getComboOption = (chartData, payload) => {
            const { x_axis, y_axes } = payload;
            if (!y_axes[0] || !y_axes[1]) return {};
            return { color: palettes[settings.activePalette], tooltip: { trigger: 'axis', axisPointer: { type: 'cross' }, formatter: _createTooltipFormatter }, legend: { data: y_axes }, xAxis: [{ type: 'category', data: [...new Set(chartData.map(d => d[x_axis]))], axisPointer: { type: 'shadow' } }], yAxis: [{ type: 'value', name: y_axes[0], axisLabel: { formatter: formatNumber } }, { type: 'value', name: y_axes[1], position: 'right', axisLabel: { formatter: formatNumber } }], series: [{ name: y_axes[0], type: 'bar', yAxisIndex: 0, data: chartData.map(d => d[y_axes[0]] || 0), label: { show: settings.showDataLabels, position: 'top', formatter: p => formatNumber(p.value) } }, { name: y_axes[1], type: 'line', yAxisIndex: 1, data: chartData.map(d => d[y_axes[1]] || 0), label: { show: settings.showDataLabels, position: 'top', formatter: p => formatNumber(p.value) } }], grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true } };
        };
        const _getPieOption = (chartData, payload) => ({ color: palettes[settings.activePalette], tooltip: { trigger: 'item', formatter: p => `${p.name}: ${formatNumber(p.value)} (${p.percent}%)` }, series: [{ name: payload.y_axes[0], type: 'pie', radius: '60%', data: chartData.map(d => ({ value: d[payload.y_axes[0]], name: d[payload.x_axis] })), emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.5)' } } }] });
        const _getTreemapOption = (chartData, payload) => ({ color: palettes[settings.activePalette], tooltip: { formatter: p => `${p.name}: ${formatNumber(p.value)}` }, series: [{ type: 'treemap', data: chartData.map(d => ({ name: d[payload.x_axis], value: d[payload.y_axes[0]] })) }] });
        const _getHeatmapOption = (chartData, payload) => {
            const { heatmap_x, heatmap_y, heatmap_value } = payload;
            if (!heatmap_x || !heatmap_y || !heatmap_value) return {};
            const xData = [...new Set(dataState.records.map(d => d[heatmap_x]))].sort();
            const yData = [...new Set(dataState.records.map(d => d[heatmap_y]))].sort();
            const seriesData = chartData.map(d => [xData.indexOf(d[heatmap_x]), yData.indexOf(d[heatmap_y]), d[heatmap_value]]).filter(p => p[0] >= 0 && p[1] >= 0);
            return { tooltip: { position: 'top', formatter: p => formatNumber(p.value[2]) }, grid: { height: '80%', top: '10%' }, xAxis: { type: 'category', data: xData, splitArea: { show: true } }, yAxis: { type: 'category', data: yData, splitArea: { show: true } }, visualMap: { min: Math.min(...seriesData.map(d => d[2])), max: Math.max(...seriesData.map(d => d[2])), inRange: { color: ['#e0f3ff', '#69c0ff', '#003a8c'] }, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%', formatter: formatNumber }, series: [{ name: 'Heatmap Data', type: 'heatmap', data: seriesData, label: { show: true, formatter: p => formatNumber(p.value[2]) }, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }] };
        };
        const _getScatterOption = (chartData, payload) => ({ color: palettes[settings.activePalette], tooltip: { trigger: 'item', formatter: p => `${payload.x_axis}: ${formatNumber(p.value[0])}<br/>${payload.y_axes[0]}: ${formatNumber(p.value[1])}` }, xAxis: { type: 'value', name: payload.x_axis, axisLabel: { formatter: formatNumber } }, yAxis: { type: 'value', name: payload.y_axes[0], axisLabel: { formatter: formatNumber } }, series: [{ symbolSize: 10, type: 'scatter', data: chartData.map(d => [d[payload.x_axis], d[payload.y_axes[0]]]) }], grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true } });
        const _getMapOption = (chartData, payload) => {
            const { geo_field, value_field } = payload;
            if (!geo_field || !value_field) return {};
            return { tooltip: { trigger: 'item', formatter: p => p.data ? `${p.name}<br/>${value_field}: ${formatNumber(p.value)}` : p.name }, visualMap: { left: 'left', min: Math.min(...chartData.map(d => d[value_field])), max: Math.max(...chartData.map(d => d[value_field])), inRange: { color: ['#e0f3ff', '#69c0ff', '#003a8c'] }, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%', formatter: formatNumber }, series: [{ name: 'Map Data', type: 'map', map: 'world', roam: true, emphasis: { label: { show: true }, itemStyle: { areaColor: '#ee6666' } }, data: chartData.map(d => ({ name: getMapCountryName(d[geo_field]), value: d[value_field] })) }] };
        };
        const _getBoxPlotOption = (chartData, payload) => {
            const { categories, boxplotData } = chartData;
            return {
                color: palettes[settings.activePalette],
                tooltip: {
                    trigger: 'item',
                    axisPointer: { type: 'shadow' }
                },
                grid: { left: '10%', right: '10%', bottom: '15%' },
                xAxis: {
                    type: 'category',
                    data: categories,
                    boundaryGap: true,
                    nameGap: 30,
                    splitArea: { show: false },
                    axisLabel: {
                        formatter: '{value}'
                    },
                    splitLine: { show: false }
                },
                yAxis: {
                    type: 'value',
                    name: payload.y_axes[0],
                    splitArea: { show: true }
                },
                series: [
                    {
                        name: 'BoxPlot',
                        type: 'boxplot',
                        data: boxplotData,
                    }
                ]
            };
        };

        const _getSankeyOption = (chartData, payload) => {
            return {
                tooltip: {
                    trigger: 'item',
                    triggerOn: 'mousemove'
                },
                series: [{
                    type: 'sankey',
                    data: chartData.nodes,
                    links: chartData.links,
                    emphasis: {
                        focus: 'adjacency'
                    },
                    lineStyle: {
                        color: 'gradient',
                        curveness: 0.5
                    }
                }]
            };
        };
        const _getWordCloudOption = (chartData, payload) => {
            return {
                tooltip: {
                    show: true
                },
                series: [{
                    type: 'wordCloud',
                    sizeRange: [12, 60],
                    rotationRange: [-90, 90],
                    rotationStep: 45,
                    gridSize: 8,
                    shape: 'circle',
                    width: '80%',
                    height: '80%',
                    textStyle: {
                        color: () => {
                            return 'rgb(' + [
                                Math.round(Math.random() * 160),
                                Math.round(Math.random() * 160),
                                Math.round(Math.random() * 160)
                            ].join(',') + ')';
                        }
                    },
                    data: chartData
                }]
            };
        };
        const _getGanttOption = (chartData, payload) => {
            return {
                tooltip: {
                    trigger: 'item',
                    formatter: (params) => {
                        const [start, end] = params.value.slice(1);
                        return `${params.name}<br/>Start: ${new Date(start).toLocaleDateString()}<br/>End: ${new Date(end).toLocaleDateString()}`;
                    }
                },
                xAxis: {
                    type: 'time',
                    min: chartData.startTime,
                },
                yAxis: {
                    type: 'category',
                    data: chartData.categories,
                    splitLine: { show: true }
                },
                series: [{
                    type: 'custom',
                    renderItem: (params, api) => {
                        const categoryIndex = api.value(0);
                        const start = api.coord([api.value(1), categoryIndex]);
                        const end = api.coord([api.value(2), categoryIndex]);
                        const height = api.size([0, 1])[1] * 0.6;
                        const rectShape = echarts.graphic.clipRectByRect({
                            x: start[0],
                            y: start[1] - height / 2,
                            width: end[0] - start[0],
                            height: height
                        }, {
                            x: params.coordSys.x,
                            y: params.coordSys.y,
                            width: params.coordSys.width,
                            height: params.coordSys.height
                        });
                        return rectShape && {
                            type: 'rect',
                            shape: rectShape,
                            style: api.style()
                        };
                    },
                    itemStyle: {
                        opacity: 0.8
                    },
                    encode: {
                        x: [1, 2],
                        y: 0
                    },
                    data: chartData.seriesData
                }]
            };
        };

        const createSnapshot = () => {
            return JSON.parse(JSON.stringify({
                worksheets: worksheets.value,
                dashboardLayout: dashboardLayout.value,
                calculatedFields: dataState.calculatedFields,
                dimensions: dataState.dimensions,
                measures: dataState.measures,
            }));
        };
        
        const saveStateSnapshot = () => {
            if (isTimeTraveling) return; 
            if (historyIndex.value < history.length - 1) {
                history.splice(historyIndex.value + 1);
            }
            history.push(createSnapshot());
            historyIndex.value = history.length - 1;
        };
        
        const loadStateFromSnapshot = (snapshot) => {
            isTimeTraveling = true;
            worksheets.value = snapshot.worksheets.map(w => ({ ...w, shelves: reactive(w.shelves), analytics: reactive(w.analytics || { showTrendLine: false, forecastPeriods: 3 }) }));
            dashboardLayout.value = snapshot.dashboardLayout;
            dataState.calculatedFields = snapshot.calculatedFields;
            dataState.dimensions = snapshot.dimensions;
            dataState.measures = snapshot.measures;
            if (!worksheets.value.find(w => w.id === activeWorksheetId.value)) {
                activeWorksheetId.value = worksheets.value[0]?.id || null;
            }
            nextTick(() => {
                const updatePromise = uiState.viewMode === 'worksheet'
                    ? updateVisualization()
                    : renderDashboardCharts();
                updatePromise.finally(() => {
                    isTimeTraveling = false;
                });
            });
        };
        
        const undo = () => {
            if (canUndo.value) {
                historyIndex.value--;
                loadStateFromSnapshot(history[historyIndex.value]);
            }
        };
        
        const redo = () => {
            if (canRedo.value) {
                historyIndex.value++;
                loadStateFromSnapshot(history[historyIndex.value]);
            }
        };
        
        const classifyFields = (data) => {
            if (!data || data.length === 0) return { dimensions: [], measures: [] };
            const headers = Object.keys(data[0] || {});
            const sample = data[0];
            return headers.reduce((acc, h) => {
                const fieldType = typeof sample[h] === 'number' ? 'measure' : 'dimension';
                acc[fieldType === 'measure' ? 'measures' : 'dimensions'].push({ name: h, type: fieldType });
                return acc;
            }, { dimensions: [], measures: [] });
        };
        
        const loadNewData = (parsedResults, fileName) => {
            resetApplication(true);
            dataState.records = parsedResults.data;
            dataState.file = { name: fileName };
            const classified = classifyFields(dataState.records);
            dataState.dimensions = classified.dimensions;
            dataState.measures = classified.measures;
            history.splice(0, history.length); 
            saveStateSnapshot();
        };
        
        const handleFileUpload = (event) => {
            const selectedFile = event.target.files[0];
            if (!selectedFile) return;
            setLoading(true);
            dataState.file = selectedFile;
            if (selectedFile.name.endsWith('.datacanvas')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const workspace = JSON.parse(e.target.result);
                        settings.activePalette = workspace.activePalette || 'default';
                        settings.activeNumberFormat = workspace.activeNumberFormat || 'default';
                        settings.autoUpdateEnabled = workspace.autoUpdateEnabled !== false;
                        loadStateFromSnapshot(workspace);
                    } catch (err) {
                        alert('Failed to load workspace file. It may be corrupted.');
                        console.error("Error loading workspace:", err);
                    } finally {
                        setLoading(false);
                    }
                };
                reader.readAsText(selectedFile);
            } else {
                Papa.parse(selectedFile, {
                    header: true,
                    dynamicTyping: true,
                    skipEmptyLines: true,
                    complete: (results) => {
                        loadNewData(results, selectedFile.name);
                        setLoading(false);
                    },
                    error: (err) => {
                        alert(`Error parsing CSV file: ${err.message}`);
                        setLoading(false);
                    }
                });
            }
            event.target.value = '';
        };
        
        const loadDataFromUrl = async () => {
            const url = dataState.dataUrl.trim();
            if (!url) return alert('Please enter a URL.');
            setLoading(true);
            try {
                const response = await fetch(url);
                if (!response.ok) throw new Error(`Network response was not ok (${response.status})`);
                const csvText = await response.text();
                Papa.parse(csvText, {
                    header: true,
                    dynamicTyping: true,
                    skipEmptyLines: true,
                    complete: (results) => {
                        const fileName = url.split('/').pop();
                        loadNewData(results, fileName);
                        dataState.dataUrl = '';
                    },
                    error: (err) => {
                        alert(`Error parsing CSV from URL: ${err.message}`);
                    }
                });
            } catch (error) {
                alert(`Failed to fetch or process data from URL: ${error.message}`);
            } finally {
                setLoading(false);
            }
        };
        
        const saveWorkspace = () => {
            const workspace = createSnapshot();
            const blob = new Blob([JSON.stringify(workspace, null, 2)], { type: 'application/json' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'workspace.datacanvas';
            link.click();
            URL.revokeObjectURL(link.href);
        };
        
        const exportAsPNG = () => {
            if (!mainChart) return alert('No chart to export.');
            const dataUrl = mainChart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: settings.theme === 'light' ? '#fff' : '#18181b' });
            const link = document.createElement('a');
            link.href = dataUrl;
            link.download = `${activeWorksheet.value.name || 'datacanvas-chart'}.png`;
            link.click();
            uiState.isDropdownVisible = false;
        };
        
        const exportDataAsCSV = () => {
            const dataToExport = activeWorksheet.value?.chartData;
            if (!dataToExport || dataToExport.length === 0) return alert('No data to export.');
            const csv = Papa.unparse(dataToExport);
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.setAttribute("download", `${activeWorksheet.value.name}-data.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            uiState.isDropdownVisible = false;
        };
        
        const resetApplication = (skipConfirm = false, shouldSaveState = true) => {
            if (skipConfirm || confirm("Are you sure? This will clear all data and worksheets.")) {
                dataState.records = [];
                dataState.dimensions = [];
                dataState.measures = [];
                dataState.calculatedFields = [];
                dataState.file = null;
                dashboardLayout.value = [];
                dashboardCharts.clear();
                Object.keys(dashboardFilters).forEach(key => delete dashboardFilters[key]);
                const newSheet = { id: 1, name: 'Sheet 1', shelves: reactive({ columns: [], rows: [], color: [], filters: [] }), activeChartType: 'bar', analytics: reactive({ showTrendLine: false, forecastPeriods: 3, model: 'linear' }), chartData: [] };
                worksheets.value = [newSheet];
                activeWorksheetId.value = 1;
                worksheetCounter = 1;
                if (mainChart) mainChart.clear();
                activeFilter.value = null;
                modals.settings = false;
                if (shouldSaveState) {
                    history.splice(0, history.length);
                    historyIndex.value = -1;
                    nextTick(() => saveStateSnapshot()); 
                }
            }
        };
        
        const addWorksheet = () => {
            worksheetCounter = worksheets.value.reduce((maxId, sheet) => Math.max(sheet.id, maxId), 0) + 1;
            const newSheet = {
                id: worksheetCounter,
                name: `Sheet ${worksheetCounter}`,
                shelves: reactive({ columns: [], rows: [], color: [], filters: [] }),
                activeChartType: 'bar',
                analytics: reactive({ showTrendLine: false, forecastPeriods: 3, model: 'linear' }),
                chartData: []
            };
            worksheets.value.push(newSheet);
            activeWorksheetId.value = newSheet.id;
        };
        
        const removeWorksheet = (idToRemove) => {
            const index = worksheets.value.findIndex(w => w.id === idToRemove);
            if (index === -1) return;
            if (activeWorksheetId.value === idToRemove) {
                if (worksheets.value.length > 1) {
                    const newActiveIndex = (index === 0) ? 0 : index - 1;
                    const nextSheet = worksheets.value[newActiveIndex === index ? index + 1 : newActiveIndex];
                    activeWorksheetId.value = nextSheet.id;
                } else {
                    activeWorksheetId.value = null;
                }
            }
            worksheets.value.splice(index, 1);
        };
        
        const setActiveWorksheet = (id) => {
            activeWorksheetId.value = id;
        };
        
        const setChartType = (type) => {
            if (activeWorksheet.value) {
                activeWorksheet.value.activeChartType = type;
            }
        };
        
        const openCalcFieldModal = () => {
            newCalcField.name = '';
            newCalcField.formula = '';
            modals.calculatedField = true;
        };
        
        const closeCalcFieldModal = () => {
            modals.calculatedField = false;
        };
        
        const submitCalcField = () => {
            if (!newCalcField.name.trim() || !newCalcField.formula.trim()) {
                return alert('Please provide a name and a formula.');
            }
            const functions = ['SUM', 'AVG', 'COUNT', 'COUNTD', 'MIN', 'MAX', 'MEDIAN', 'STDEV', 'VAR'];
            const regex = new RegExp(`(${functions.join('|')})\\(\\[(.*?)\\]\\)`, 'g');
            if (!regex.test(newCalcField.formula)) {
                 return alert(`Formula must contain at least one valid function, like: ${functions.join(', ')}.`);
            }
            
            const createdField = {
                name: newCalcField.name.trim(),
                type: 'measure',
                isCalculated: true,
                formula: newCalcField.formula.trim()
            };
            dataState.calculatedFields.push(createdField);
            dataState.measures.push(createdField);
            closeCalcFieldModal();
        };
        
        const removeFromShelf = (shelf, item) => {
            if (!activeWorksheet.value) return;
            const shelfArray = shelf === 'filters' ? activeWorksheet.value.shelves.filters : activeWorksheet.value.shelves[shelf];
            const findIndexFn = shelf === 'filters' ? f => f.field === item.field : i => i.name === item.name;
            const index = shelfArray.findIndex(findIndexFn);
            if (index > -1) {
                shelfArray.splice(index, 1);
            }
            if (shelf === 'filters' && activeFilter.value?.field === item.field) {
                activeFilter.value = null;
            }
            handleShelfUpdate();
        };
        
        const resetView = () => {
            if (activeWorksheet.value) {
                Object.assign(activeWorksheet.value.shelves, { columns: [], rows: [], color: [], filters: [] });
            }
            activeFilter.value = null;
            requestVisualizationUpdate();
            uiState.isDropdownVisible = false;
        };
        
        const editFilter = (field) => {
            const activeSheet = activeWorksheet.value;
            if (!activeSheet) return;
            const existingFilter = activeSheet.shelves.filters.find(f => f.field === field.name);
            if (existingFilter) {
                activeFilter.value = existingFilter;
                return;
            }
            if (field.type === 'dimension') {
                const uniqueValues = [...new Set(dataState.records.map(r => r[field.name]))].sort();
                const newFilter = reactive({
                    field: field.name,
                    filter_type: 'dimension',
                    mode: 'list',
                    n: 10,
                    by: dataState.measures[0]?.name || '',
                    values: [...uniqueValues],
                    uniqueValues: uniqueValues,
                });
                activeSheet.shelves.filters.push(newFilter);
                activeFilter.value = newFilter;
            } else {
                const values = dataState.records.map(r => r[field.name]);
                const newFilter = reactive({
                    field: field.name,
                    filter_type: 'range',
                    values: {
                        min: Math.min(...values),
                        max: Math.max(...values)
                    }
                });
                activeSheet.shelves.filters.push(newFilter);
                activeFilter.value = newFilter;
            }
        };

        const handleAddFilter = (event) => {
            const field = event.added.element;
            
            const tempIndex = event.added.newIndex;
            activeWorksheet.value.shelves.filters.splice(tempIndex, 1);
            
            editFilter(field);
        };
        
        const showContextMenu = (field, event) => {
            uiState.contextMenu = {
                visible: true,
                top: event.clientY,
                left: event.clientX,
                field: field
            };
        };
        
        const convertFieldType = () => {
            const field = uiState.contextMenu.field;
            if (!field) return;
            if (field.type === 'measure') {
                dataState.measures = dataState.measures.filter(m => m.name !== field.name);
                dataState.dimensions.push({ ...field, type: 'dimension' });
            } else {
                dataState.dimensions = dataState.dimensions.filter(d => d.name !== field.name);
                dataState.measures.push({ ...field, type: 'measure' });
            }
            uiState.contextMenu.visible = false;
        };
        
        const toggleDropdown = () => {
            uiState.isDropdownVisible = !uiState.isDropdownVisible;
        };
        
        const openSettings = () => {
            modals.settings = true;
            uiState.isDropdownVisible = false;
        };
        
        const openGroupModal = () => {
            const fieldToGroup = uiState.contextMenu.field;
            if (!fieldToGroup) return;
            groupingState.field = fieldToGroup.name;
            groupingState.uniqueValues = [...new Set(dataState.records.map(r => r[fieldToGroup.name]))];
            groupingState.selectedValues = [];
            groupingState.newGroupName = '';
            modals.grouping = true;
            uiState.contextMenu.visible = false;
        };
        
        const submitGroup = () => {
            if (!groupingState.newGroupName.trim() || groupingState.selectedValues.length === 0) {
                return alert('Please provide a group name and select values.');
            }
            const fieldName = groupingState.field;
            const newFieldName = `${fieldName} (Group)`;
            dataState.records.forEach(record => {
                if (groupingState.selectedValues.includes(record[fieldName])) {
                    record[newFieldName] = groupingState.newGroupName;
                } else {
                    record[newFieldName] = record[fieldName];
                }
            });
            if (!dataState.dimensions.some(d => d.name === newFieldName)) {
                dataState.dimensions.push({ name: newFieldName, type: 'dimension' });
            }
            modals.grouping = false;
            requestVisualizationUpdate();
        };
        
        const toggleSort = (field) => {
            const allShelfFields = [...activeWorksheet.value.shelves.columns, ...activeWorksheet.value.shelves.rows];
            allShelfFields.forEach(f => {
                if (f.name !== field.name) {
                    delete f.sort;
                }
            });
            if (!field.sort) {
                field.sort = 'desc';
            } else if (field.sort === 'desc') {
                field.sort = 'asc';
            } else {
                delete field.sort;
            }
            requestVisualizationUpdate();
        };
        
        const drillDate = (field, direction) => {
            const currentIndex = dateLevels.indexOf(field.drillLevel);
            if (direction === 'down' && currentIndex < dateLevels.length - 1) {
                field.drillLevel = dateLevels[currentIndex + 1];
            } else if (direction === 'up' && currentIndex > 0) {
                field.drillLevel = dateLevels[currentIndex - 1];
            }
            requestVisualizationUpdate();
        };
        
        const handleDashboardChartClick = (params, sourceWorksheet) => {
            const sourceId = sourceWorksheet.id.toString();
            const field = sourceWorksheet.shelves.columns.find(f => f.type === 'dimension')?.name;
            const value = params.name;
            if (!field) return;
            if (dashboardFilters[sourceId]?.value === value) {
                delete dashboardFilters[sourceId];
            } else {
                dashboardFilters[sourceId] = { field, value, sourceId };
            }
            renderDashboardCharts();
        };
        
        const startEdit = (worksheet) => {
            uiState.editingId = worksheet.id;
            uiState.editText = worksheet.name;
            nextTick(() => {
                const input = document.getElementById(`editor-${worksheet.id}`);
                input?.focus();
                input?.select();
            });
        };
        
        const saveEdit = (worksheet) => {
            if (uiState.editText.trim()) {
                worksheet.name = uiState.editText.trim();
            }
            uiState.editingId = null;
        };
        
        const cancelEdit = () => {
            uiState.editingId = null;
        };
        
        const addToDashboard = () => {
            const sheet = activeWorksheet.value;
            if (!sheet || dashboardLayout.value.some(item => item.i === sheet.id.toString())) {
                return alert('This sheet is already on the dashboard or no sheet is active.');
            }
            const y = dashboardLayout.value.reduce((maxY, item) => Math.max(item.y + item.h, maxY), 0);
            dashboardLayout.value.push({
                x: 0, y: y, w: 6, h: 5, i: sheet.id.toString(), worksheetId: sheet.id
            });
            uiState.viewMode = 'dashboard';
        };
        
        const removeFromDashboard = (id) => {
            const index = dashboardLayout.value.findIndex(item => item.i === id);
            if (index > -1) {
                dashboardLayout.value.splice(index, 1);
            }
            const chartId = parseInt(id.toString());
            const chartInstance = dashboardCharts.get(chartId);
            if (chartInstance) {
                chartInstance.dispose();
                dashboardCharts.delete(chartId);
            }
        };
        
        const gridItemStyle = (item) => ({
            gridColumn: `${item.x + 1} / span ${item.w}`,
            gridRow: `${item.y + 1} / span ${item.h}`,
        });
        
        const dragStart = (item, event) => {
            if (event.target.classList.contains('resize-handle')) return;
            const gridRect = dashboardGridRef.value.getBoundingClientRect();
            gridInteraction.active = true;
            gridInteraction.type = 'drag';
            gridInteraction.item = item;
            gridInteraction.startX = event.clientX;
            gridInteraction.startY = event.clientY;
            gridInteraction.initialX = item.x;
            gridInteraction.initialY = item.y;
            gridInteraction.gridCellWidth = (gridRect.width - (11 * 10)) / 12;
            gridInteraction.gridCellHeight = 50 + 10;
            window.addEventListener('mousemove', mouseMove);
            window.addEventListener('mouseup', mouseUp);
        };
        
        const resizeStart = (item, event) => {
            const gridRect = dashboardGridRef.value.getBoundingClientRect();
            gridInteraction.active = true;
            gridInteraction.type = 'resize';
            gridInteraction.item = item;
            gridInteraction.startX = event.clientX;
            gridInteraction.startY = event.clientY;
            gridInteraction.initialW = item.w;
            gridInteraction.initialH = item.h;
            gridInteraction.gridCellWidth = (gridRect.width - (11 * 10)) / 12;
            gridInteraction.gridCellHeight = 50 + 10;
            window.addEventListener('mousemove', mouseMove);
            window.addEventListener('mouseup', mouseUp);
        };
        
        const mouseMove = (event) => {
            if (!gridInteraction.active) return;
            if (gridInteraction.animationFrameId) cancelAnimationFrame(gridInteraction.animationFrameId);
            
            gridInteraction.animationFrameId = requestAnimationFrame(() => {
                event.preventDefault();
                const dx = event.clientX - gridInteraction.startX;
                const dy = event.clientY - gridInteraction.startY;
                if (gridInteraction.type === 'drag') {
                    const newX = gridInteraction.initialX + Math.round(dx / gridInteraction.gridCellWidth);
                    const newY = gridInteraction.initialY + Math.round(dy / gridInteraction.gridCellHeight);
                    gridInteraction.item.x = Math.max(0, Math.min(newX, 12 - gridInteraction.item.w));
                    gridInteraction.item.y = Math.max(0, newY);
                } else if (gridInteraction.type === 'resize') {
                    const newW = gridInteraction.initialW + Math.round(dx / gridInteraction.gridCellWidth);
                    const newH = gridInteraction.initialH + Math.round(dy / gridInteraction.gridCellHeight);
                    gridInteraction.item.w = Math.max(2, Math.min(newW, 12 - gridInteraction.item.x));
                    gridInteraction.item.h = Math.max(2, newH);
                }
            });
        };
        
        const mouseUp = () => {
            if (!gridInteraction.active) return;
            if (gridInteraction.animationFrameId) cancelAnimationFrame(gridInteraction.animationFrameId);
            gridInteraction.active = false;
            nextTick(() => {
                dashboardCharts.forEach(chart => chart.resize());
            });
            window.removeEventListener('mousemove', mouseMove);
            window.removeEventListener('mouseup', mouseUp);
        };

        const openAnalysisModal = () => {
            analysisState.result = null;
            analysisState.error = '';
            analysisState.measure1 = dataState.measures[0]?.name || '';
            analysisState.measure2 = dataState.measures[1]?.name || '';
            analysisState.ttest.measure = dataState.measures[0]?.name || '';
            analysisState.ttest.dimension = dataState.dimensions[0]?.name || '';
            analysisState.zscore.measure = dataState.measures[0]?.name || '';
            modals.statisticalAnalysis = true;
        };

        const runAnalysis = async () => {
            analysisState.result = null;
            analysisState.error = '';
            setLoading(true);

            try {
                const payload = {
                    testType: analysisState.testType,
                    records: JSON.parse(JSON.stringify(dataState.records)),
                    params: {}
                };

                switch (analysisState.testType) {
                    case 'correlation':
                        if (!analysisState.measure1 || !analysisState.measure2) throw new Error('Please select two measures.');
                        payload.params = { measure1: analysisState.measure1, measure2: analysisState.measure2 };
                        break;
                    case 'ttest':
                        if (!analysisState.ttest.measure || !analysisState.ttest.dimension) throw new Error('Please select a measure and a dimension.');
                        payload.params = { measure: analysisState.ttest.measure, dimension: analysisState.ttest.dimension };
                        break;
                    case 'zscore':
                        if (!analysisState.zscore.measure) throw new Error('Please select a measure.');
                        payload.params = { measure: analysisState.zscore.measure };
                        break;

                    case 'clustering':
                        if (analysisState.clustering.fields.length < 2) throw new Error('Please select at least two measures to cluster.');
                        payload.params = { k: analysisState.clustering.k, fields: analysisState.clustering.fields};
                        break;
                    default:
                        throw new Error('Invalid test type selected.');
                }
                
                const workerResult = await new Promise((resolve, reject) => {
                    const tempWorker = new Worker('worker.js');
                    tempWorker.onmessage = (event) => {
                        tempWorker.terminate();
                        resolve(event.data);
                    };
                    tempWorker.onerror = (error) => {
                        tempWorker.terminate();
                        reject(error);
                    };
                    tempWorker.postMessage({ type: 'runAnalysis', payload });
                });

                if (workerResult.error) {
                    analysisState.error = workerResult.error;
                } else {
                    if (analysisState.testType === 'clustering') {
                        // For clustering, we update the main dataset and close the modal
                        dataState.records = workerResult.records;
                        if (!dataState.dimensions.some(d => d.name === 'Cluster')) {
                            dataState.dimensions.push({ name: 'Cluster', type: 'dimension' });
                        }
                        modals.statisticalAnalysis = false;
                        alert('Clustering complete! A new "Cluster" dimension has been added.');
                        requestVisualizationUpdate(); // Redraw the chart
                    } else {
                        // For other analyses, we just show the results in the modal
                        analysisState.result = workerResult.result;
                    }
                }
            } catch (e) {
                console.error("Analysis failed:", e);
                analysisState.error = e.message;
            } finally {
                setLoading(false);
            }
        };

        const openBinningModal = () => { // <-- ADDED: New function to open the binning modal
            binningState.measure = dataState.measures[0]?.name || '';
            binningState.binSize = 100;
            binningState.binName = `${binningState.measure || 'Field'} Bins`;
            modals.binning = true;
        };

        const submitBin = async () => { // <-- ADDED: New function to submit the binning request
            if (!binningState.binName.trim() || !binningState.measure || !binningState.binSize) {
                return alert('Please fill in all binning fields.');
            }
            if (dataState.dimensions.some(d => d.name === binningState.binName) || dataState.measures.some(m => m.name === binningState.binName)) {
                return alert('Bin name already exists. Please choose a different name.');
            }

            setLoading(true);
            try {
                const workerResult = await new Promise((resolve, reject) => {
                    const tempWorker = new Worker('worker.js');
                    tempWorker.onmessage = (event) => {
                        tempWorker.terminate();
                        resolve(event.data);
                    };
                    tempWorker.onerror = (error) => {
                        tempWorker.terminate();
                        reject(error);
                    };
                    tempWorker.postMessage({ 
                        type: 'runBinning', 
                        payload: {
                            records: JSON.parse(JSON.stringify(dataState.records)),
                            measure: binningState.measure,
                            binSize: binningState.binSize,
                            binName: binningState.binName
                        }
                    });
                });

                if (workerResult.error) {
                    throw new Error(workerResult.error);
                }
                
                // Add the new binned field to the dimensions
                dataState.dimensions.push({ name: binningState.binName, type: 'dimension' });
                // Replace the old records with the new binned records
                dataState.records = workerResult.records;

                modals.binning = false;
                alert(`Successfully created bin field "${binningState.binName}"!`);

            } catch (e) {
                console.error("Binning failed:", e);
                alert(`Failed to create bins: ${e.message}`);
            } finally {
                setLoading(false);
            }
        };


        watch(() => settings.theme, (newTheme) => { localStorage.setItem('datacanvas-theme', newTheme); document.documentElement.classList.toggle('dark', newTheme === 'dark'); if (uiState.viewMode === 'dashboard') renderDashboardCharts(); else updateVisualization(); }, { immediate: true });
        watch([() => settings.activePalette, () => settings.activeNumberFormat, () => settings.showDataLabels, () => settings.autoUpdateEnabled], () => { localStorage.setItem('datacanvas-palette', settings.activePalette); localStorage.setItem('datacanvas-format', settings.activeNumberFormat); localStorage.setItem('datacanvas-labels', settings.showDataLabels); localStorage.setItem('datacanvas-autoupdate', settings.autoUpdateEnabled); if (settings.autoUpdateEnabled && uiState.isDirty) applyManualUpdate(); else if (uiState.viewMode === 'worksheet') updateVisualization(); else renderDashboardCharts(); });
        watch(
            () => ({
                worksheets: worksheets.value,
                dashboardLayout: dashboardLayout.value,
                calculatedFields: dataState.calculatedFields,
                dimensions: dataState.dimensions,
                measures: dataState.measures,
            }),
            saveStateSnapshot,
            { deep: true }
        );
        watch([() => uiState.viewMode, activeWorksheetId], ([newMode]) => { if (newMode === 'dashboard') { nextTick(renderDashboardCharts); } else { nextTick(() => { updateVisualization(); if (mainChart) setTimeout(() => mainChart.resize(), 50); }); } });
        watch(activeFilter, () => { requestVisualizationUpdate(); }, { deep: true });
        watch(() => activeWorksheet.value?.activeChartType, (newType, oldType) => { if (newType && newType !== oldType && uiState.viewMode === 'worksheet') { requestVisualizationUpdate(); } });

        onMounted(() => {
            try { worker = new Worker('worker.js'); } catch (e) { console.error("Failed to create worker.", e); }
            resetApplication(true);
            if (chartContainer.value) {
                const resizeObserver = new ResizeObserver(() => mainChart?.resize());
                resizeObserver.observe(chartContainer.value);
            }
            window.addEventListener('resize', () => dashboardCharts.forEach(chart => chart.resize()));
            window.addEventListener('click', (event) => {
                if (uiState.contextMenu.visible) uiState.contextMenu.visible = false;
                if (uiState.isDropdownVisible && dropdownContainer.value && !dropdownContainer.value.contains(event.target)) { uiState.isDropdownVisible = false; }
            });
        });

        return {
            chartContainer, dashboardGridRef, dropdownContainer,
            dataState, uiState, modals, settings, analysisState, canRunAnalysis,
            binningState, openBinningModal, submitBin,
            worksheets, activeWorksheetId, dashboardLayout, activeFilter, newCalcField, groupingState, gridInteraction,
            activeWorksheet, activeShelves, activeChartType, chartConfigured, isDarkTheme, canUndo, canRedo, isAnalyticsPanelVisible, chartSuggestions, palettes, numberFormats,
            handleFileUpload,
            saveWorkspace, exportAsPNG, exportDataAsCSV, loadDataFromUrl, resetApplication, addWorksheet, removeWorksheet, setActiveWorksheet,
            startEdit, saveEdit, cancelEdit, addToDashboard, removeFromDashboard, showContextMenu, convertFieldType, openGroupModal, submitGroup, openSettings,
            resetView, toggleDropdown, openCalcFieldModal, closeCalcFieldModal, submitCalcField, handleShelfUpdate, getWorksheetById, gridItemStyle, dragStart, resizeStart, getSortClass, toggleSort,
            drillDate, handleDashboardChartClick, applyManualUpdate, undo, redo, getTableCellStyle, formatNumber, setChartType, editFilter, removeFromShelf,
            openAnalysisModal, handleAddFilter, runAnalysis,
        };
    }
})

app.component('draggable', vuedraggable);
app.mount('#app');

chartSuggestions