namespace YodaTransformer;

public static class MathOps
{
    public static float[][] MatMul(float[][] A, float[][] B)
    {
        int m = A.Length;
        int k = A[0].Length;
        int n = B[0].Length;
        float[][] C = new float[m][];
        for (int i = 0; i < m; i++)
        {
            C[i] = new float[n];
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int p = 0; p < k; p++)
                    sum += A[i][p] * B[p][j];
                C[i][j] = sum;
            }
        }
        return C;
    }

    public static float[][] Transpose(float[][] M)
    {
        int m = M.Length;
        int n = M[0].Length;
        float[][] T = new float[n][];
        for (int i = 0; i < n; i++)
        {
            T[i] = new float[m];
            for (int j = 0; j < m; j++)
                T[i][j] = M[j][i];
        }
        return T;
    }

    public static float[] Softmax(float[] v)
    {
        int len = v.Length;
        float max = v[0];
        for (int i = 1; i < len; i++)
            if (v[i] > max) max = v[i];

        float[] result = new float[len];
        float sum = 0f;
        for (int i = 0; i < len; i++)
        {
            result[i] = MathF.Exp(v[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < len; i++)
            result[i] /= sum;
        return result;
    }

    public static float[][] SoftmaxRows(float[][] M)
    {
        int rows = M.Length;
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++)
            result[i] = Softmax(M[i]);
        return result;
    }

    public static float[] LayerNorm(float[] v)
    {
        int len = v.Length;
        float mean = 0f;
        for (int i = 0; i < len; i++)
            mean += v[i];
        mean /= len;

        float variance = 0f;
        for (int i = 0; i < len; i++)
        {
            float diff = v[i] - mean;
            variance += diff * diff;
        }
        variance /= len;

        float std = MathF.Sqrt(variance + 1e-6f);
        float[] result = new float[len];
        for (int i = 0; i < len; i++)
            result[i] = (v[i] - mean) / std;
        return result;
    }

    public static float[][] LayerNormRows(float[][] M)
    {
        int rows = M.Length;
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++)
            result[i] = LayerNorm(M[i]);
        return result;
    }

    public static float[] ReLU(float[] v)
    {
        int len = v.Length;
        float[] result = new float[len];
        for (int i = 0; i < len; i++)
            result[i] = v[i] > 0f ? v[i] : 0f;
        return result;
    }

    public static float[] AddVectors(float[] a, float[] b)
    {
        int len = a.Length;
        float[] result = new float[len];
        for (int i = 0; i < len; i++)
            result[i] = a[i] + b[i];
        return result;
    }

    public static float[][] AddMatrices(float[][] A, float[][] B)
    {
        int rows = A.Length;
        int cols = A[0].Length;
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            result[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                result[i][j] = A[i][j] + B[i][j];
        }
        return result;
    }

    public static float[][] ScalarMul(float[][] M, float s)
    {
        int rows = M.Length;
        int cols = M[0].Length;
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            result[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                result[i][j] = M[i][j] * s;
        }
        return result;
    }

    public static float[] MatVecMul(float[][] M, float[] v)
    {
        int m = M.Length;
        int n = v.Length;
        float[] result = new float[m];
        for (int i = 0; i < m; i++)
        {
            float sum = 0f;
            for (int j = 0; j < n; j++)
                sum += M[i][j] * v[j];
            result[i] = sum;
        }
        return result;
    }

    public static float[] VecMatMul(float[] v, float[][] M)
    {
        int m = v.Length;
        int n = M[0].Length;
        float[] result = new float[n];
        for (int j = 0; j < n; j++)
        {
            float sum = 0f;
            for (int i = 0; i < m; i++)
                sum += v[i] * M[i][j];
            result[j] = sum;
        }
        return result;
    }

    public static float ClipNorm(float[] grad, float maxNorm)
    {
        float norm = 0f;
        for (int i = 0; i < grad.Length; i++)
            norm += grad[i] * grad[i];
        norm = MathF.Sqrt(norm);
        return norm > maxNorm ? maxNorm / norm : 1f;
    }

    // Box-Muller transform to sample from standard normal distribution.
    private static float NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    public static float[][] RandomMatrix(int rows, int cols, Random rng, float scale)
    {
        float[][] M = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            M[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                M[i][j] = NextGaussian(rng) * scale;
        }
        return M;
    }

    public static float[] RandomVector(int size, Random rng, float scale)
    {
        float[] v = new float[size];
        for (int i = 0; i < size; i++)
            v[i] = NextGaussian(rng) * scale;
        return v;
    }
}
