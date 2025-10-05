package moa.classifiers.trees;

import java.io.Serializable;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Statistics;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

public class StochasticGradientTree extends AbstractClassifier implements MultiClassClassifier, Regressor {
    private static final long serialVersionUID = 1L;

    protected Node root;
    protected Discretizer[] discretizers;

    public ClassOption discretizerOption = new ClassOption("discretizer", 'd',
            "Discretizer to use for numeric attributes.", Discretizer.class, EqualWidthDiscretizer.class.getName() + " -bins 64");

    public IntOption gracePeriodOption = new IntOption("gracePeriod", 'g',
            "The number of instances a leaf should observe between split attempts.", 200, 1, Integer.MAX_VALUE);

    public FloatOption lambdaOption = new FloatOption("lambda", 'l',
            "Regularization parameter lambda; should be interpreted as the inverse of a learning rate.", 0.1, 0.0, Float.MAX_VALUE);

    public FloatOption confidenceOption = new FloatOption("confidence", 'c',
            "The allowable error in split decision, a value closer to 0 takes longer to decide.", 1E-7, 0.0, 1.0);

    public FloatOption minImprovementOption = new FloatOption("minImprovement", 'm',
            "The minimum loss reduction required to allow a split to occur.", 1e-3, 0.0, Float.MAX_VALUE);

    public MultiChoiceOption splitTestOption = new MultiChoiceOption("splitTest", 's',
            "Which type of hypothesis test to use for determining when to split.",
            new String[] { "First"},
            new String[] { "Choose the first split with a significant improvement." },
            0);

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        // TODO Auto-generated method stub
        return new Measurement[0];
    }

    @Override
    public void resetLearningImpl() {
        root = null;
        discretizers = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (root == null) {
            if (inst.classAttribute().isNominal()) {
                root = new Node(new double[inst.classAttribute().numValues()]);
            } else {
                root = new Node(new double[1]);
            }
        }

        observeDiscretizers(inst);

        // Find the leaf node for the instance
        Node leaf = root.findLeaf(inst);

        Gradients grads = computeGradients(inst, leaf.getValues());

        // Update the leaf node with the instance
        leaf.updateStats(inst, grads);

        // Check if the leaf node should be split
        if (leaf.getCount() % gracePeriodOption.getValue() == 0) {
            Split split = leaf.findBestSplit();

            if (splitTestOption.getChosenIndex() == 0) {
                double pValue = computeTTestPValue(split, leaf.getCount());

                if (pValue < confidenceOption.getValue() && split.deltaLoss < -minImprovementOption.getValue()) {
                    leaf.applySplit(split);
                }
            } else {
                throw new IllegalStateException("Unknown split test option: " + splitTestOption.getChosenIndex());
            }
        }
    }

    protected double computeTTestPValue(Split split, int instances) {
        // H0: the expected loss is zero
        // HA: the expected loss is not zero

        try {
            double F = instances * Math.pow(split.deltaLoss, 2.0) / split.deltaLossVar;

            return Statistics.FProbability(F, 1, instances - 1);
        }
        catch(ArithmeticException e) {
            return 1.0;
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (inst.classAttribute().isNominal()) {
            if (root == null) {
                return new double[inst.classAttribute().numValues()];
            } else {
                Node leaf = root.findLeaf(inst);
                double[] values = leaf.getValues();
                softmax(values);
                return values;
            }
        } else {
            if (root == null) {
                return new double[] { 0.0 };
            } else {
                Node leaf = root.findLeaf(inst);
                return leaf.getValues();
            }
        }
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    protected void observeDiscretizers(Instance inst) {
        if (discretizers == null) {
            discretizers = new Discretizer[inst.numAttributes()];
            for (int i = 0; i < inst.numAttributes(); i++) {
                if (i != inst.classIndex() && inst.attribute(i).isNumeric()) {
                    discretizers[i] = (Discretizer) getPreparedClassOption(discretizerOption);
                }
            }
        }

        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i != inst.classIndex() && inst.attribute(i).isNumeric()) {
                discretizers[i].observe(inst.value(i));
            }
        }
    }

    protected void softmax(double[] vec) {
        double max = Double.NEGATIVE_INFINITY;

        for (double v : vec) {
            if (v > max) {
                max = v;
            }
        }

        double sum = 0.0;

        for (int i = 0; i < vec.length; i++) {
            vec[i] = Math.exp(vec[i] - max);
            sum += vec[i];
        }

        for (int i = 0; i < vec.length; i++) {
            vec[i] /= sum;
        }
    }

    protected Gradients computeGradients(Instance inst, double[] predictions) {
        // Figure out what type of learning problem we've got going on here
        if (inst.classAttribute().isNominal()) {
            Gradients grads = new Gradients(inst.classAttribute().numValues());

            predictions = predictions.clone();
            softmax(predictions);
            
            int label = (int) inst.classValue();

            for (int i = 0; i < predictions.length; i++) {
                double prob = predictions[i];
                double indicator = (i == label) ? 1.0 : 0.0;
                grads.gradients[i] = prob - indicator;
                grads.hessians[i] = prob * (1.0 - prob);
            }

            return grads;
        } else {
            // Regression case: single gradient and hessian of squared loss
            Gradients grads = new Gradients(1);
            grads.gradients[0] = predictions[0] - inst.classValue();
            grads.hessians[0] = 1.0;

            return grads;
        }
    }

    public abstract static class Discretizer extends AbstractOptionHandler {
        public IntOption binsOption = new IntOption("bins", 'b',
            "Number of bins for discretization.", 64, 1, Integer.MAX_VALUE);

        abstract void observe(double value);
        abstract int getBin(double value);

        @Override
        public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        }
    }

    public static class EqualWidthDiscretizer extends Discretizer {
        private static final long serialVersionUID = 1L;

        private double min;
        private double max;
        private boolean initialized = false;
        
        @Override
        public void observe(double value) {
            if (!initialized) {
                min = value;
                max = value;
                initialized = true;
            } else {
                if (value < min) {
                    min = value;
                }

                if (value > max) {
                    max = value;
                }
            }
        }

        @Override
        public int getBin(double value) {
            int numBins = binsOption.getValue();

            if (!initialized || numBins <= 0 || min == max) {
                return 0;
            }

            double binWidth = (max - min) / numBins;
            int bin = (int) ((value - min) / binWidth);

            if (bin < 0) {
                return 0;
            }

            if (bin >= numBins) {
                return numBins - 1;
            }

            return bin;
        }

        @Override
        public String getPurposeString() {
            return "Discretizer that uses equal-width bins.";
        }

        @Override
        public Discretizer copy() {
            EqualWidthDiscretizer copy = new EqualWidthDiscretizer();
            copy.binsOption = (IntOption) this.binsOption.copy();
            copy.min = this.min;
            copy.max = this.max;
            copy.initialized = this.initialized;
            return copy;
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
            String pad = new String(new char[indent]).replace('\0', ' ');
            sb.append(pad).append(this.getClass().getName()).append("\n");
            sb.append(pad).append("Bins: ").append(binsOption.getValue()).append("\n");
            sb.append(pad).append("Min: ").append(min).append("\n");
            sb.append(pad).append("Max: ").append(max).append("\n");
        }
    }

    protected static class Gradients implements Serializable {
        private static final long serialVersionUID = 1L;

        public double[] gradients;
        public double[] hessians;

        public Gradients(int size) {
            this.gradients = new double[size];
            this.hessians = new double[size];
        }

        public void add(Gradients other) {
            if (this.gradients.length != other.gradients.length) {
                throw new IllegalArgumentException("Gradient sizes do not match.");
            }

            for (int i = 0; i < this.gradients.length; i++) {
                this.gradients[i] += other.gradients[i];
                this.hessians[i] += other.hessians[i];
            }
        }

        public void subtract(Gradients other) {
            if (this.gradients.length != other.gradients.length) {
                throw new IllegalArgumentException("Gradient sizes do not match.");
            }

            for (int i = 0; i < this.gradients.length; i++) {
                this.gradients[i] -= other.gradients[i];
                this.hessians[i] -= other.hessians[i];
            }
        }

        public Gradients clone() {
            Gradients copy = new Gradients(this.gradients.length);
            System.arraycopy(this.gradients, 0, copy.gradients, 0, this.gradients.length);
            System.arraycopy(this.hessians, 0, copy.hessians, 0, this.hessians.length);
            return copy;
        }
    }

    protected static class GradientStats implements Serializable {
        private static final long serialVersionUID = 1L;

        private int count;
        private Gradients sum;
        private Gradients scaledVariance;
        private double[] scaledCovariance;

        public GradientStats(int size) {
            this.count = 0;
            this.sum = new Gradients(size);
            this.scaledVariance = new Gradients(size);
            this.scaledCovariance = new double[size];
        }
        
        public void add(Gradients grads) {
            if (grads.gradients.length != sum.gradients.length) {
                throw new IllegalArgumentException("Gradient sizes do not match.");
            }

            Gradients oldMean = getMean();
            sum.add(grads);
            count++;
            Gradients newMean = getMean();

            for (int i = 0; i < grads.gradients.length; i++) {
                scaledVariance.gradients[i] += (grads.gradients[i] - oldMean.gradients[i]) * (grads.gradients[i] - newMean.gradients[i]);
                scaledVariance.hessians[i] += (grads.hessians[i] - oldMean.hessians[i]) * (grads.hessians[i] - newMean.hessians[i]);
                scaledCovariance[i] += (grads.gradients[i] - oldMean.gradients[i]) * (grads.hessians[i] - newMean.hessians[i]);
            }
        }

        public void add(GradientStats other) {
            if (other.sum.gradients.length != this.sum.gradients.length) {
                throw new IllegalArgumentException("Gradient sizes do not match.");
            }

            if (other.count == 0) {
                return;
            }

            if (this.count == 0) {
                this.count = other.count;
                this.sum = other.sum.clone();
                this.scaledVariance = other.scaledVariance.clone();
                this.scaledCovariance = other.scaledCovariance.clone();
                return;
            }

            Gradients meanDiff = other.getMean();
            meanDiff.subtract(this.getMean());

            for (int i = 0; i < sum.gradients.length; i++) {
                this.scaledVariance.gradients[i] += other.scaledVariance.gradients[i] + (this.count * other.count * meanDiff.gradients[i] * meanDiff.gradients[i]) / (this.count + other.count);
                this.scaledVariance.hessians[i] += other.scaledVariance.hessians[i] + (this.count * other.count * meanDiff.hessians[i] * meanDiff.hessians[i]) / (this.count + other.count);
                this.scaledCovariance[i] += other.scaledCovariance[i] + (this.count * other.count * meanDiff.gradients[i] * meanDiff.hessians[i]) / (this.count + other.count);
            }

            sum.add(other.sum);
            this.count += other.count;
        }

        public Gradients getMean() {
            if (count == 0) {
                return new Gradients(sum.gradients.length);
            }

            Gradients mean = new Gradients(sum.gradients.length);

            for (int i = 0; i < sum.gradients.length; i++) {
                mean.gradients[i] = sum.gradients[i] / count;
                mean.hessians[i] = sum.hessians[i] / count;
            }

            return mean;
        }

        public Gradients getVariance() {
            if (count <= 1) {
                Gradients ret = new Gradients(sum.gradients.length);

                for (int i = 0; i < sum.gradients.length; i++) {
                    ret.gradients[i] = Double.POSITIVE_INFINITY;
                    ret.hessians[i] = Double.POSITIVE_INFINITY;
                }
            }

            Gradients variance = new Gradients(sum.gradients.length);

            for (int i = 0; i < sum.gradients.length; i++) {
                variance.gradients[i] = scaledVariance.gradients[i] / (count - 1);
                variance.hessians[i] = scaledVariance.hessians[i] / (count - 1);
            }

            return variance;
        }

        public double[] getCovariance() {
            double[] covariance = new double[scaledCovariance.length];

            if (count <= 1) {
                for (int i = 0; i < scaledCovariance.length; i++) {
                    covariance[i] = Double.POSITIVE_INFINITY;
                }
            } else {
                for (int i = 0; i < scaledCovariance.length; i++) {
                    covariance[i] = scaledCovariance[i] / (count - 1);
                }
            }
            
            return covariance;
        }

        public int getCount() {
            return count;
        }
    }

    protected static class Split implements Serializable {
        private static final long serialVersionUID = 1L;

        public int attributeIndex;
        public int threshold;
        public boolean isNominal;
        public double deltaLoss;
        public double deltaLossVar;
        public double[][] deltaValues;
    }

    protected class Node implements Serializable {
        private static final long serialVersionUID = 1L;

        private double[] values;
        private Node[] children;
        private int splitAttributeIndex;
        private int splitThreshold;
        private GradientStats totalStats;
        private GradientStats[][] attributeStats;

        public Node(double[] values) {
            this.values = values;
            this.children = null;
            this.totalStats = null;
            this.attributeStats = null;
        }

        public Node findLeaf(Instance inst) {
            if (children == null) {
                return this;
            } else {
                if (inst.attribute(splitAttributeIndex).isNominal()) {
                    int attrValue = (int) inst.value(splitAttributeIndex);
                    return children[attrValue].findLeaf(inst);
                } else {
                    int attrValue = discretizers[splitAttributeIndex].getBin(inst.value(splitAttributeIndex));

                    if (attrValue <= splitThreshold) {
                        return children[0].findLeaf(inst);
                    } else {
                        return children[1].findLeaf(inst);
                    }
                }
            }
        }

        public void updateStats(Instance inst, Gradients grads) {
            if (totalStats == null) {
                totalStats = new GradientStats(values.length);
                attributeStats = new GradientStats[inst.numAttributes()][];

                for (int i = 0; i < inst.numAttributes(); i++) {
                    if (i != inst.classIndex()) {
                        if (inst.attribute(i).isNominal()) {
                            attributeStats[i] = new GradientStats[inst.attribute(i).numValues()];

                            for (int j = 0; j < inst.attribute(i).numValues(); j++) {
                                attributeStats[i][j] = new GradientStats(values.length);
                            }
                        } else {
                            int numBins = discretizers[i].binsOption.getValue();
                            attributeStats[i] = new GradientStats[numBins];

                            for (int j = 0; j < numBins; j++) {
                                attributeStats[i][j] = new GradientStats(values.length);
                            }
                        }
                    }
                }
            }

            totalStats.add(grads);

            for (int i = 0; i < inst.numAttributes(); i++) {
                if (i != inst.classIndex()) {
                    int bin;

                    if (inst.attribute(i).isNominal()) {
                        bin = (int) inst.value(i);
                    } else {
                        bin = discretizers[i].getBin(inst.value(i));
                    }

                    attributeStats[i][bin].add(grads);
                }
            }
        }

        protected double combineMean(double mean1, int count1, double mean2, int count2) {
            if (count1 + count2 == 0) {
                return 0.0;
            }

            return (mean1 * count1 + mean2 * count2) / (count1 + count2);
        }

        protected double combineVariance(double mean1, double var1, int count1, double mean2, double var2, int count2) {
            if (count1 == 0) {
                return var2;
            }

            if (count2 == 0) {
                return var1;
            }
            
            double m = combineMean(mean1, count1, mean2, count2);

            double s1 = var1 * (count1 - 1);
            double s2 = var2 * (count2 - 1);

            double t1 = s1 + count1 * Math.pow(mean1, 2);
            double t2 = s2 + count2 * Math.pow(mean2, 2);
            double t = t1 + t2;

            double s = t / (count1 + count2) - m;
            return ((double)(count1 + count2) / (count1 + count2 - 1)) * s;
        }

        protected double[] computeDeltaPreds(GradientStats stats) {
            double[] deltaPreds = new double[stats.sum.gradients.length];
            Gradients mean = stats.getMean();

            for (int i = 0; i < deltaPreds.length; i++) {
                deltaPreds[i] = -mean.gradients[i] / (mean.hessians[i] + lambdaOption.getValue());
            }

            return deltaPreds;
        }

        protected double computeLossMean(GradientStats stats, double[] deltaPreds) {
            double loss = 0.0;
            Gradients mean = stats.getMean();

            for (int i = 0; i < mean.gradients.length; i++) {
                loss += mean.gradients[i] * deltaPreds[i]
                    + 0.5 * mean.hessians[i] * Math.pow(deltaPreds[i], 2);
            }

            return loss;
        }

        protected double computeLossVar(GradientStats stats, double[] deltaPreds) {
            double lossVar = 0.0;
            Gradients vars = stats.getVariance();
            double[] covs = stats.getCovariance();

            for (int i = 0; i < deltaPreds.length; i++) {
                lossVar += Math.pow(deltaPreds[i], 2) * vars.gradients[i]
                    + 0.25 * Math.pow(deltaPreds[i], 4) * vars.hessians[i]
                    + Math.pow(deltaPreds[i], 3) * covs[i];
            }

            return lossVar;
        }

        public Split findBestSplit() {
            Split best = new Split();
            best.attributeIndex = -1;
            best.deltaValues = new double[][] { computeDeltaPreds(totalStats) };
            best.deltaLoss = computeLossMean(totalStats, best.deltaValues[0]);
            best.deltaLossVar = computeLossVar(totalStats, best.deltaValues[0]);
            
            for (int i = 0; i < attributeStats.length; i++) {
                if (attributeStats[i] == null) {
                    // This attribute is the class attribute
                    continue;
                }

                Split candidate = new Split();
                candidate.attributeIndex = i;

                if (discretizers[i] == null) {
                    candidate.isNominal = true;
                    candidate.deltaValues = new double[attributeStats[i].length][values.length];
                    int obs = 0;

                    for (int j = 0; j < attributeStats[i].length; j++) {
                        candidate.deltaValues[j] = computeDeltaPreds(attributeStats[i][j]);
                        double leafMean = computeLossMean(attributeStats[i][j], candidate.deltaValues[j]);
                        double leafVar = computeLossVar(attributeStats[i][j], candidate.deltaValues[j]);
                        int leafObs = attributeStats[i][j].getCount();

                        candidate.deltaLoss = combineMean(candidate.deltaLoss, obs, leafMean, leafObs);
                        candidate.deltaLossVar = combineVariance(candidate.deltaLoss, candidate.deltaLossVar, obs, leafMean, leafVar, leafObs);
                        obs += leafObs;
                    }
                } else {
                    candidate.isNominal = false;
                    candidate.deltaValues = new double[2][];

                    GradientStats[] forwardCumulative = new GradientStats[attributeStats[i].length - 1];
                    GradientStats[] backwardCumulative = new GradientStats[attributeStats[i].length - 1];

                    for (int j = 0; j < forwardCumulative.length; j++) {
                        forwardCumulative[j] = new GradientStats(values.length);
                        forwardCumulative[j].add(attributeStats[i][j]);

                        if (j > 0) {
                            forwardCumulative[j].add(forwardCumulative[j - 1]);
                        }
                    }

                    for (int j = backwardCumulative.length - 1; j >= 0; j--) {
                        backwardCumulative[j] = new GradientStats(values.length);
                        backwardCumulative[j].add(attributeStats[i][j + 1]);

                        if (j + 1 < backwardCumulative.length) {
                            backwardCumulative[j].add(backwardCumulative[j + 1]);
                        }
                    }

                    candidate.deltaLoss = Double.POSITIVE_INFINITY;

                    for (int j = 0; j < forwardCumulative.length; j++) {
                        double[] leftDeltaValues = computeDeltaPreds(forwardCumulative[j]);
                        double leftMean = computeLossMean(forwardCumulative[j], leftDeltaValues);
                        double leftVar = computeLossVar(forwardCumulative[j], leftDeltaValues);
                        int leftObs = forwardCumulative[j].getCount();
                        
                        double[] rightDeltaValues = computeDeltaPreds(backwardCumulative[j]);
                        double rightMean = computeLossMean(backwardCumulative[j], rightDeltaValues);
                        double rightVar = computeLossVar(backwardCumulative[j], rightDeltaValues);
                        int rightObs = backwardCumulative[j].getCount();
                        
                        double splitMean = combineMean(leftMean, leftObs, rightMean, rightObs);
                        double splitVar = combineVariance(leftMean, leftVar, leftObs, rightMean, rightVar, rightObs);

                        if (splitMean < candidate.deltaLoss) {
                            candidate.deltaLoss = splitMean;
                            candidate.deltaLossVar = splitVar;
                            candidate.threshold = j;
                            candidate.deltaValues[0] = leftDeltaValues;
                            candidate.deltaValues[1] = rightDeltaValues;
                        }
                    }
                }

                if (candidate.deltaLoss < best.deltaLoss) {
                    best = candidate;
                }
            }

            return best;
        }

        public void applySplit(Split split) {
            if (split.attributeIndex == -1) {
                totalStats = null;
                
                for (int i = 0; i < values.length; i++) {
                    values[i] += split.deltaValues[0][i];
                }

                return;
            }

            this.splitAttributeIndex = split.attributeIndex;
            this.splitThreshold = split.threshold;

            if (split.isNominal) {
                this.children = new Node[split.deltaValues.length];

                for (int i = 0; i < split.deltaValues.length; i++) {
                    double[] newValues = new double[values.length];

                    for (int j = 0; j < values.length; j++) {
                        newValues[j] = values[j] + split.deltaValues[i][j];
                    }

                    this.children[i] = new Node(newValues);
                }
            } else {
                this.children = new Node[2];

                double[] leftValues = new double[values.length];
                double[] rightValues = new double[values.length];

                for (int j = 0; j < values.length; j++) {
                    leftValues[j] = values[j] + split.deltaValues[0][j];
                    rightValues[j] = values[j] + split.deltaValues[1][j];
                }

                this.children[0] = new Node(leftValues);
                this.children[1] = new Node(rightValues);
            }

            this.totalStats = null;
            this.attributeStats = null;
        }

        public int getCount() {
            return totalStats == null ? 0 : totalStats.getCount();
        }

        public double[] getValues() {
            return values.clone();
        }
    }
}
