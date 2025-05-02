import torch

# Adapted from the paper's code

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError

class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError

def get_task_sampler(
    task_name, n_dims, batch_size, n_pivots=None, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "piecewise_linear_regression": PiecewiseLinearRegression,
        "piecewise_linear_vector_regression": PiecewiseLinearVectorRegression,
        "piecewise_linear_vector_regression_multi_pivot": PiecewiseLinearVectorRegressionMultiPivot
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        if n_pivots is not None:
            return lambda **args: task_cls(n_dims, batch_size, n_pivots, pool_dict, **args, **kwargs)
        else:
            return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError

class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class PiecewiseLinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """
        Implements a piecewise linear function:
            f(x) = a*x + b if x < c, and f(x) = d*x + e otherwise.
        Each task instance gets its own set of parameters.

        scale: a constant factor to scale the predictions.
        """
        super(PiecewiseLinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        # Generate parameters in one of three ways:
        # 1. If neither pool_dict nor seeds is provided, sample parameters normally.
        #    We store parameters as a tensor of shape (batch_size, 5),
        #    where each row is [a, b, c, d, e].
        if pool_dict is None and seeds is None:
            self.params = torch.randn(self.b_size, 5)
        # 2. If seeds are provided, generate parameters reproducibly.
        elif seeds is not None:
            self.params = torch.zeros(self.b_size, 5)
            generator = torch.Generator()
            assert len(seeds) == self.b_size, "Number of seeds must match batch_size."
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.params[i] = torch.randn(5, generator=generator)
        # 3. Otherwise, use a pre-generated pool of parameters.
        else:
            assert "params" in pool_dict, "Expected key 'params' in pool_dict."
            indices = torch.randperm(len(pool_dict["params"]))[:batch_size]
            self.params = pool_dict["params"][indices]

    def evaluate(self, xs_b):
        """
        xs_b: a tensor of input values of shape (batch_size, n_points, n_dims).
            For scalar inputs, n_dims should equal 1.
        Returns:
            A tensor of outputs of shape (batch_size, n_points).
        """
        # Remove the last dimension which is 1 for scalar inputs.
        xs_b = xs_b.squeeze(-1)  # now shape is (batch_size,)

        # Unpack parameters: each is of shape (batch_size, 1)
        a = self.params[:, 0].unsqueeze(1)
        b = self.params[:, 1].unsqueeze(1)
        c = self.params[:, 2].unsqueeze(1)
        d = self.params[:, 3].unsqueeze(1)
        e = self.params[:, 4].unsqueeze(1)

        # Compute f(x) elementwise using the piecewise logic.
        # torch.where operates elementwise on the tensors.
        f_x = torch.where(xs_b < c, a * xs_b + b, d * xs_b + e)
        
        # Apply scaling and return
        return self.scale * f_x

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        """
        Generate a pool of parameters to be shared between tasks.
        Each task gets a vector of 5 parameters drawn from a standard normal distribution.
        Returns:
            A dictionary with key "params" and a tensor of shape (num_tasks, 5).
        """
        return {"params": torch.randn(num_tasks, 5)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class PiecewiseLinearVectorRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """
        Implements a piecewise linear function for vector inputs:
            f(x) = a*x+b if w*x < c and f(x)= d*x+e  otherwise.
        Here: a, d, w are vectors; b, c, e are scalars.
        scale: a constant factor to scale the predictions.
        """
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        # Total parameters per task: 3 vectors each of length n_dims, plus scalars b, c, e
        total = 3 * n_dims + 3
        if pool_dict is None and seeds is None:
            self.params = torch.randn(batch_size, total)
        elif seeds is not None:
            self.params = torch.zeros(batch_size, total)
            gen = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                gen.manual_seed(seed)
                self.params[i] = torch.randn(total, generator=gen)
        else:
            assert "params" in pool_dict, "Expected key 'params' in pool_dict."
            indices = torch.randperm(len(pool_dict["params"]))[:batch_size]
            self.params = pool_dict["params"][indices]

    def evaluate(self, xs_b):
        """
        xs_b: a tensor of input values of shape (batch_size, n_points, n_dims).
        Returns:
            tensor of shape (batch_size, n_points).
        """
        # a, d, w: shape (batch_size, 1, n_dims)
        a = self.params[:, :self.n_dims].unsqueeze(1)
        d = self.params[:, self.n_dims:2*self.n_dims].unsqueeze(1)
        w = self.params[:, 2*self.n_dims:3*self.n_dims].unsqueeze(1)
        # b, c, e: shape (batch_size, 1, 1)
        b = self.params[:, 3*self.n_dims].view(-1,1,1)
        c = self.params[:, 3*self.n_dims+1 ].view(-1,1,1)
        e = self.params[:, 3*self.n_dims+2 ].view(-1,1,1)

        # Compute dot products
        # dot_w : shape (B, P, 1)
        dot_w = (w * xs_b).sum(dim=-1, keepdim=True)
        mask = dot_w < c
        dot_a = (a * xs_b).sum(dim=-1, keepdim=True)
        dot_d = (d * xs_b).sum(dim=-1, keepdim=True)

        # Piecewise linear scalar output
        f_x = torch.where(mask, dot_a + b, dot_d + e)
        # Remove last dim to get (batch_size, n_points)
        return (self.scale * f_x).squeeze(-1)

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        total = 3 * n_dims + 3
        return {"params": torch.randn(num_tasks, total)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    


class PiecewiseLinearVectorRegressionMultiPivot(Task):
    def __init__(self, n_dims, batch_size, n_pivots=2, pool_dict=None, seeds=None, scale=1):
        """
        Implements a piecewise linear function for vector inputs with multiple pivot points:
            f(x) = a_1*x + b_1 if w*x < c_1
            f(x) = a_2*x + b_2 if c_1 <= w*x < c_2
            ...
            f(x) = a_k*x + b_k if w*x >= c_{k-1}
        Here: a_i, w are vectors; b_i, c_i are scalars.
        n_pivots: number of pivot points (resulting in n_pivots+1 pieces)
        scale: a constant factor to scale the predictions.
        """
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.n_pivots = n_pivots
        
        # Total parameters per task: 
        # - (n_pivots + 1) vectors a_i of length n_dims for each piece
        # - 1 vector w of length n_dims
        # - (n_pivots + 1) scalars b_i for each piece
        # - n_pivots scalars c_i for pivot points
        total = (n_pivots + 1) * n_dims + n_dims + (n_pivots + 1) + n_pivots

        if pool_dict is None and seeds is None:
            self.params = torch.randn(batch_size, total)
            # Sort the pivot points
            pivot_start = (n_pivots + 2) * n_dims + (n_pivots + 1)
            pivot_end = pivot_start + n_pivots
            self.params[:, pivot_start:pivot_end], _ = torch.sort(self.params[:, pivot_start:pivot_end], dim=1)
        elif seeds is not None:
            self.params = torch.zeros(batch_size, total)
            gen = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                gen.manual_seed(seed)
                params = torch.randn(total, generator=gen)
                # Sort the pivot points
                pivot_start = (n_pivots + 2) * n_dims + (n_pivots + 1)
                pivot_end = pivot_start + n_pivots
                params[pivot_start:pivot_end], _ = torch.sort(params[pivot_start:pivot_end])
                self.params[i] = params
        else:
            assert "params" in pool_dict, "Expected key 'params' in pool_dict."
            indices = torch.randperm(len(pool_dict["params"]))[:batch_size]
            self.params = pool_dict["params"][indices]

    def evaluate(self, xs_b):
        """
        xs_b: a tensor of input values of shape (batch_size, n_points, n_dims).
        Returns:
            tensor of shape (batch_size, n_points).
        """
        batch_size = xs_b.shape[0]
        n_points = xs_b.shape[1]

        # Extract parameters
        # w: shape (batch_size, 1, n_dims)
        w = self.params[:, self.n_dims*(self.n_pivots+1):self.n_dims*(self.n_pivots+2)].unsqueeze(1)
        
        # Compute w*x for all points
        # dot_w: shape (batch_size, n_points, 1)
        dot_w = (w * xs_b).sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        f_x = torch.zeros(batch_size, n_points, 1, device=xs_b.device)
        
        # Handle first piece (w*x < c_1)
        a = self.params[:, :self.n_dims].unsqueeze(1)
        b = self.params[:, (self.n_pivots+2)*self.n_dims].view(-1, 1, 1)
        c = self.params[:, (self.n_pivots+2)*self.n_dims + self.n_pivots + 1].view(-1, 1, 1)
        # mask_scalar = (dot_w < c).repeat(1, 1, self.n_dims)
        mask = (dot_w < c)
        b = b.repeat(1, n_points, 1)
        # print(b.shape)
        # print(mask_scalar.shape)
        # print(mask.shape)
        # print((a * xs_b).shape)
        # print((a * xs_b).sum(dim=-1, keepdim=True).shape)
        f_x[mask] = (a * xs_b).sum(dim=-1, keepdim=True)[mask] + b[mask]
        
        # Handle middle pieces
        for i in range(1, self.n_pivots):
            a = self.params[:, i*self.n_dims:(i+1)*self.n_dims].unsqueeze(1)
            b = self.params[:, (self.n_pivots+2)*self.n_dims + i].view(-1, 1, 1)
            b = b.repeat(1, n_points, 1)
            c_prev = self.params[:, (self.n_pivots+2)*self.n_dims + self.n_pivots + i].view(-1, 1, 1)
            c_next = self.params[:, (self.n_pivots+2)*self.n_dims + self.n_pivots + i + 1].view(-1, 1, 1)
            mask = (dot_w >= c_prev) & (dot_w < c_next)
            f_x[mask] = (a * xs_b).sum(dim=-1, keepdim=True)[mask] + b[mask]
        
        # Handle last piece (w*x >= c_k)
        a = self.params[:, self.n_pivots*self.n_dims:(self.n_pivots+1)*self.n_dims].unsqueeze(1)
        b = self.params[:, (self.n_pivots+2)*self.n_dims + self.n_pivots].view(-1, 1, 1)
        b = b.repeat(1, n_points, 1)
        c = self.params[:, -1].view(-1, 1, 1)
        mask = dot_w >= c
        f_x[mask] = (a * xs_b).sum(dim=-1, keepdim=True)[mask] + b[mask]
        
        # Remove last dim to get (batch_size, n_points)
        return (self.scale * f_x).squeeze(-1)

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, n_pivots=2, **kwargs):
        total = (n_pivots + 1) * n_dims + n_dims + (n_pivots + 1) + n_pivots
        params = torch.randn(num_tasks, total)
        # Sort the pivot points
        pivot_start = (n_pivots + 2) * n_dims + (n_pivots + 1)
        pivot_end = pivot_start + n_pivots
        params[:, pivot_start:pivot_end], _ = torch.sort(params[:, pivot_start:pivot_end], dim=1)
        return {"params": params}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

########################################################################
# Example usage:
########################################################################

if __name__ == "__main__":
    torch.manual_seed(282)
    n_dims = 1
    batch_size = 5
    num_tasks = 100     # For generating a pool (if using pool_dict).
    scale = 1           # Scaling factor for the function output.
    task_factory = get_task_sampler(
        "piecewise_linear_regression", n_dims, batch_size, num_tasks=num_tasks, scale=scale
    )
    piecewise_task = task_factory()
    xs = torch.randn(batch_size, n_dims)
    outputs = piecewise_task.evaluate(xs)
    print("Input xs:", xs)
    print("Output f(x):", outputs)

