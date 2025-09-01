// worker.js

importScripts('https://cdn.jsdelivr.net/npm/simple-statistics@7.8.3/dist/simple-statistics.min.js');

const isGeoField = (fieldName) => {
    const lowerCaseName = fieldName.toLowerCase();
    const geoKeywords = ['country', 'nation', 'region', 'state', 'province', 'location'];
    return geoKeywords.some(keyword => lowerCaseName.includes(keyword));
};

const isDateField = (fieldName) => {
    return fieldName.toLowerCase().includes('date');
};

const getDerivedDateValue = (dateString, level) => {
    const date = new Date(dateString);
    if (isNaN(date)) return 'Invalid Date';

    const year = date.getFullYear();
    switch (level) {
        case 'year':
            return year;
        case 'quarter':
            const quarter = Math.floor(date.getMonth() / 3) + 1;
            return `${year}-Q${quarter}`;
        case 'month':
            const month = (date.getMonth() + 1).toString().padStart(2, '0');
            return `${year}-${month}`;
        default:
            return dateString;
    }
};

const aggregate = (func, values) => {
    const numericValues = values.map(Number).filter(v => !isNaN(v));
    if (numericValues.length === 0) {
        return ['SUM', 'AVG', 'COUNT'].includes(func) ? 0 : null;
    }

    switch (func) {
        case 'SUM':
            return numericValues.reduce((sum, val) => sum + val, 0);
        case 'AVG':
            const sum = numericValues.reduce((s, v) => s + v, 0);
            return sum / numericValues.length;
        case 'COUNT':
            return values.length;
        case 'COUNTD':
            return new Set(values).size;
        case 'MIN':
            return Math.min(...numericValues);
        case 'MAX':
            return Math.max(...numericValues);
        case 'MEDIAN':
            numericValues.sort((a, b) => a - b);
            const mid = Math.floor(numericValues.length / 2);
            return numericValues.length % 2 !== 0 ? numericValues[mid] : (numericValues[mid - 1] + numericValues[mid]) / 2;
        case 'STDEV':
        case 'VAR':
            if (numericValues.length < 2) return 0;
            const mean = numericValues.reduce((s, v) => s + v, 0) / numericValues.length;
            const variance = numericValues.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / (numericValues.length - 1);
            return func === 'VAR' ? variance : Math.sqrt(variance);
        default:
            return null;
    }
};

const getCorrelation = (records, measure1, measure2) => {
    const values1 = records.map(r => r[measure1]).filter(v => typeof v === 'number');
    const values2 = records.map(r => r[measure2]).filter(v => typeof v === 'number');
    if (values1.length !== values2.length || values1.length < 2) return null;
    const correlation = ss.sampleCorrelation(values1, values2);
    return correlation;
};

const getTTest = (records, measure, dimension) => {
    const uniqueGroups = [...new Set(records.map(r => r[dimension]))].filter(g => g !== null).slice(0, 2);
    if (uniqueGroups.length < 2) return null;

    const group1Data = records.filter(r => r[dimension] === uniqueGroups[0]).map(r => r[measure]).filter(v => typeof v === 'number');
    const group2Data = records.filter(r => r[dimension] === uniqueGroups[1]).map(r => r[measure]).filter(v => typeof v === 'number');
    
    if (group1Data.length < 2 || group2Data.length < 2) return null;
    const ttest = ss.tTestTwoSample(group1Data, group2Data);
    
    return {
        group1: uniqueGroups[0],
        group2: uniqueGroups[1],
        tValue: ttest,
        degreesOfFreedom: group1Data.length + group2Data.length - 2,
    };
};

const getZScoreOutliers = (records, measure) => {
    const values = records.map(r => r[measure]).filter(v => typeof v === 'number');
    if (values.length < 2) return [];

    const mean = ss.mean(values);
    const standardDeviation = ss.standardDeviation(values);
    
    if (standardDeviation === 0) return [];
    
    const outliers = [];
    records.forEach((record, index) => {
        if (typeof record[measure] === 'number') {
            const zScore = (record[measure] - mean) / standardDeviation;
            if (Math.abs(zScore) > 3) {
                outliers.push({
                    row: index + 1,
                    value: record[measure],
                    zScore: zScore.toFixed(2)
                });
            }
        }
    });
    return outliers;
};

const runKMeansClustering = (records, params) => {
    const { k, fields } = params;
    
    const dataForClustering = records.map(rec => fields.map(field => rec[field]));

    const kmeans = new kMeans({ K: k });
    kmeans.cluster(dataForClustering);
    while (kmeans.step()) {
        kmeans.findClosestCentroids();
        kmeans.moveCentroids();
        if (kmeans.hasConverged()) break;
    }
    const clusterAssignments = kmeans.clusters;

    const pointToClusterMap = new Map();
    clusterAssignments.forEach((cluster, clusterIndex) => {
        cluster.forEach(point => {
            pointToClusterMap.set(JSON.stringify(point), clusterIndex + 1);
        });
    });

    const clusteredRecords = records.map((record, index) => {
        const point = dataForClustering[index];
        const clusterIndex = pointToClusterMap.get(JSON.stringify(point));
        return {
            ...record,
            'Cluster': `Cluster ${clusterIndex}`
        };
    });
    
    return { records: clusteredRecords };
};

const runStatisticalAnalysis = (payload) => {
    const { testType, records, params } = payload;
    let headers, rows;
    let result = {};

    switch(testType) {
        case 'correlation':
            const correlation = getCorrelation(records, params.measure1, params.measure2);
            headers = ['Measures', 'Correlation Coefficient'];
            rows = [[`${params.measure1} & ${params.measure2}`, correlation !== null ? correlation.toFixed(2) : 'N/A']];
            result = { headers, rows };
            break;
        case 'ttest':
            const ttest = getTTest(records, params.measure, params.dimension);
            headers = ['Measure', 'Group 1', 'Group 2', 'T-Value', 'Degrees of Freedom'];
            rows = [[params.measure, ttest?.group1, ttest?.group2, ttest?.tValue.toFixed(2), ttest?.degreesOfFreedom]];
            result = { headers, rows };
            break;
        case 'zscore':
            const outliers = getZScoreOutliers(records, params.measure);
            headers = ['Original Index', params.measure, 'Z-Score'];
            rows = outliers.map(o => [o.row, o.value, o.zScore]);
            result = { headers, rows };
            break;
        case 'clustering':
            return runKMeansClustering(records, params);
    }
    return { result };
};

const evaluateAggregateFormula = (formula, dataGroup) => {
    const functions = ['SUM', 'AVG', 'COUNT', 'COUNTD', 'MIN', 'MAX', 'MEDIAN', 'STDEV', 'VAR'];
    const regex = new RegExp(`(?<!(${functions.join('|')})\\s*\\()(\\[[^\\]]+\\])`, 'g');
    let expression = formula.replace(regex, 'SUM($2)');

    const aggRegex = new RegExp(`(${functions.join('|')})\\(\\[(.*?)\\]\\)`, 'g');
    expression = expression.replaceAll(aggRegex, (match, func, field) => {
        const values = dataGroup.map(row => row[field]).filter(v => v != null);
        return aggregate(func, values);
    });

    try {
        if (/[^0-9.+\-*/()\s]/.test(expression)) {
            throw new Error('Invalid characters detected in the final formula expression.');
        }
        return new Function('return ' + expression)();
    } catch (e) {
        console.error(`Error evaluating expression "${expression}" derived from "${formula}":`, e);
        return null;
    }
};

const processDataForChart = (data) => {
    const { records, config, measures, dashboardFilters } = data;
    if (!records || records.length === 0) {
        return { chartData: [], payload: {} };
    }

    const allFieldsOnShelves = [...config.shelves.columns, ...config.shelves.rows, ...config.shelves.color];
    const dateFieldsOnShelves = allFieldsOnShelves.filter(f => f.isDate);
    
    let processedRecords = records.map(record => {
        const newRecord = { ...record };
        dateFieldsOnShelves.forEach(field => {
            const derivedFieldName = `${field.name} (${field.drillLevel})`;
            newRecord[derivedFieldName] = getDerivedDateValue(record[field.name], field.drillLevel);
        });
        return newRecord;
    });

    const getFieldName = (field) => field?.isDate ? `${field.name} (${field.drillLevel})` : field?.name;
    const dimensionOnCol = config.shelves.columns.find(f => f.type === 'dimension');
    const measureOnCol = config.shelves.columns.find(f => f.type === 'measure');
    const colorDimension = config.shelves.color.find(f => f.type === 'dimension');
    const measuresOnRows = config.shelves.rows.filter(f => f.type === 'measure');

    const payload = {
        x_axis: getFieldName(config.activeChartType === 'scatter' ? measureOnCol : dimensionOnCol),
        y_axes: config.shelves.rows.map(r => r.name),
        color: getFieldName(colorDimension),
        filters: config.shelves.filters,
        chart_type: config.activeChartType,
        analytics: config.analytics,
    };
    
    if (payload.chart_type === 'heatmap') {
        const dims = allFieldsOnShelves.filter(f => f.type === 'dimension');
        payload.heatmap_x = getFieldName(dims[0]);
        payload.heatmap_y = getFieldName(dims[1]);
        payload.heatmap_value = measuresOnRows[0]?.name;
    }
    if (payload.chart_type === 'map') {
        const geoField = allFieldsOnShelves.find(f => f.type === 'dimension' && isGeoField(f.name));
        payload.geo_field = getFieldName(geoField);
        payload.value_field = measuresOnRows[0]?.name;
    }

    const preAggregationFilters = (payload.filters || []).filter(f => f.mode !== 'topN');
    const applicableDashboardFilters = Object.values(dashboardFilters || {})
        .filter(df => df.sourceId !== config.id.toString())
        .map(df => ({ ...df, filter_type: 'dashboard' }));
    
    const allFilters = [...preAggregationFilters, ...applicableDashboardFilters];
    if (allFilters.length > 0) {
        processedRecords = processedRecords.filter(rec =>
            allFilters.every(f => {
                const value = rec[f.field];
                switch (f.filter_type) {
                    case 'dimension': return f.values.includes(value);
                    case 'range': return value >= f.values.min && value <= f.values.max;
                    case 'dashboard': return value === f.value;
                    default: return true;
                }
            })
        );
    }

    if (payload.chart_type === 'wordCloud') {
        const textField = getFieldName(allFieldsOnShelves.find(f => f.type === 'dimension'));
        if (!textField) {
            return { chartData: [], payload };
        }
    
        const wordCounts = new Map();
        const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'in', 'of', 'to', 'is', 'it', 'for', 'with', 'on', 'as', 'by', 'that', 'this', 'i', 'you', 'he', 'she', 'we', 'they']);
    
        processedRecords.forEach(record => {
            const text = record[textField];
            if (text && typeof text === 'string') {
                const words = text.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").split(/\s+/);
    
                words.forEach(word => {
                    if (word && !stopWords.has(word)) {
                        wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
                    }
                });
            }
        });
    
        const chartData = Array.from(wordCounts.entries()).map(([name, value]) => ({ name, value }));
        chartData.sort((a, b) => b.value - a.value);
    
        return { chartData: chartData.slice(0, 100), payload };
    }

    if (payload.chart_type === 'gantt') {
        const allDims = allFieldsOnShelves.filter(f => f.type === 'dimension' && !isDateField(f.name));
        const allDates = allFieldsOnShelves.filter(f => isDateField(f.name));
    
        if (allDims.length < 1 || allDates.length < 2) {
            return { chartData: { seriesData: [], categories: [], startTime: 0 }, payload };
        }
    
        const categoryField = getFieldName(allDims[0]);
        const startDateField = getFieldName(allDates[0]);
        const endDateField = getFieldName(allDates[1]);
    
        const categories = [...new Set(processedRecords.map(r => r[categoryField]))];
        const seriesData = [];
        let startTime = Infinity;
    
        processedRecords.forEach(record => {
            const category = record[categoryField];
            const start = new Date(record[startDateField]).getTime();
            const end = new Date(record[endDateField]).getTime();
    
            if (!isNaN(start) && !isNaN(end)) {
                if (start < startTime) {
                    startTime = start;
                }
                seriesData.push({
                    name: category,
                    value: [
                        categories.indexOf(category),
                        start,
                        end
                    ]
                });
            }
        });
    
        return { chartData: { seriesData, categories, startTime }, payload };
    }

    if (payload.chart_type === 'sankey') {
        const allDims = allFieldsOnShelves.filter(f => f.type === 'dimension');
        const allMeasures = allFieldsOnShelves.filter(f => f.type === 'measure');
    
        if (allDims.length < 2 || allMeasures.length < 1) {
            return { chartData: { nodes: [], links: [] }, payload };
        }
    
        const sourceDim = getFieldName(allDims[0]);
        const targetDim = getFieldName(allDims[1]);
        const valueMeasure = allMeasures[0].name;
    
        const nodes = [...new Set([...processedRecords.map(r => r[sourceDim]), ...processedRecords.map(r => r[targetDim])])].map(name => ({ name }));
        
        const linkMap = new Map();
        processedRecords.forEach(record => {
            const source = record[sourceDim];
            const target = record[targetDim];
            const value = record[valueMeasure] || 0;
            const key = `${source}->${target}`;
    
            if (linkMap.has(key)) {
                linkMap.get(key).value += value;
            } else {
                linkMap.set(key, { source, target, value });
            }
        });
    
        const links = Array.from(linkMap.values());
        
        return { chartData: { nodes, links }, payload };
    }

    const groupingKeys = [payload.x_axis, payload.color].filter(Boolean);
    let finalData = processedRecords;

    if (payload.chart_type === 'boxplot') {
        const groups = {};
        processedRecords.forEach(record => {
            const category = record[payload.x_axis];
            if (!groups[category]) {
                groups[category] = [];
            }
            groups[category].push(record[payload.y_axes[0]]);
        });

        const categories = Object.keys(groups);
        const boxplotData = categories.map(cat => {
            const values = groups[cat].filter(v => typeof v === 'number').sort((a, b) => a - b);
            if (values.length === 0) return [0, 0, 0, 0, 0];
            return [
                ss.min(values),
                ss.quantile(values, 0.25),
                ss.median(values),
                ss.quantile(values, 0.75),
                ss.max(values)
            ];
        });
        return { chartData: { categories, boxplotData }, payload };
    }

    const needsAggregation = payload.chart_type !== 'scatter' && payload.y_axes.length > 0 && groupingKeys.length > 0;
    if (needsAggregation) {
        const grouped = new Map();
        for (const record of processedRecords) {
            const key = groupingKeys.map(k => record[k]).join('||');
            if (!grouped.has(key)) grouped.set(key, []);
            grouped.get(key).push(record);
        }

        const aggregatedResults = [];
        for (const dataGroup of grouped.values()) {
            const resultRow = {};
            groupingKeys.forEach(key => { resultRow[key] = dataGroup[0][key]; });
            
            payload.y_axes.forEach(yAxisName => {
                const field = measures.find(m => m.name === yAxisName);
                if (!field) return;

                if (field.isCalculated) {
                    resultRow[yAxisName] = evaluateAggregateFormula(field.formula, dataGroup);
                } else {
                    const aggType = field.aggregation || 'SUM';
                    const values = dataGroup.map(row => row[yAxisName]).filter(v => v != null);
                    resultRow[yAxisName] = aggregate(aggType, values);
                }
            });
            aggregatedResults.push(resultRow);
        }
        finalData = aggregatedResults;
    }
    
    const topNFilter = (payload.filters || []).find(f => f.mode === 'topN');
    if (topNFilter && finalData.length > 0 && finalData[0].hasOwnProperty(topNFilter.by)) {
        finalData.sort((a, b) => (b[topNFilter.by] || 0) - (a[topNFilter.by] || 0));
        finalData = finalData.slice(0, topNFilter.n);
    }

    if (payload.chart_type === 'line' && payload.analytics?.showTrendLine && payload.analytics?.forecastPeriods > 0) {
        const measure = payload.y_axes[0];
        if (finalData.length > 1 && measure) {
            const historicalValues = finalData.map(d => d[measure]);
            const forecastPeriods = payload.analytics.forecastPeriods;
            let forecastResult = [];
    
            if (payload.analytics.model === 'smoothing' && historicalValues.length >= 4) {
                const model = new ets(historicalValues, {
                    seasonal: 'additive',
                    period: Math.min(12, Math.floor(historicalValues.length / 2))
                });
                model.predict(forecastPeriods);
                forecastResult = model.prediction;
    
            } else {
                const regressionData = finalData.map((d, i) => [i, d[measure]]);
                const regressionLine = ss.linearRegression(regressionData);
                const trendLineFunc = ss.linearRegressionLine(regressionLine);
                
                for (let i = 0; i < forecastPeriods; i++) {
                    const x = finalData.length + i;
                    forecastResult.push(trendLineFunc(x));
                }
            }
            
            payload.analytics.forecastData = forecastResult.map(val => ({ prediction: val }));
            payload.analytics.forecastConfidence = []; 
        }
    }
    
    return { chartData: finalData, payload };
};

const runBinning = (payload) => {
    const { records, measure, binSize, binName } = payload;
    if (!measure || !binSize || !binName) {
        return { error: 'Missing required parameters for binning.' };
    }

    const newRecords = records.map(record => {
        const newRecord = { ...record };
        const value = record[measure];

        if (typeof value === 'number' && !isNaN(value)) {
            const lowerBound = Math.floor(value / binSize) * binSize;
            const upperBound = lowerBound + binSize;
            newRecord[binName] = `${lowerBound}-${upperBound}`;
        } else {
            newRecord[binName] = null;
        }
        return newRecord;
    });

    return { records: newRecords };
};

self.onmessage = (event) => {
    const { type, payload } = event.data;
    try {
        let result;
        if (type === 'runAnalysis') {
            result = runStatisticalAnalysis(payload);
        } else if (type === 'processDataForChart') {
            result = processDataForChart(payload);
        } else if (type === 'runBinning') {
            result = runBinning(payload);
        }

        self.postMessage(JSON.parse(JSON.stringify(result)));

    } catch (error) {
        console.error('Error in web worker:', error);
        self.postMessage({ error: error.message, chartData: [], payload: {} });
    }
};