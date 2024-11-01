import torch
import torch.nn as nn
import torch.nn.functional as F


class HNetLoss(nn.Module):
    def __init__(self):
        """
        H-Net Loss Module
        """
        super(HNetLoss, self).__init__()

    def forward(self, gt_pts, transformation_coefficients):
        """
        Compute H-Net Loss
        :param gt_pts: Ground truth points in the form of tensor [batch_size, num_points, 3]
        :param transformation_coefficients: Predicted transformation coefficients, shape [batch_size, 8]
        :return: Computed loss
        """
        # Define pre-defined H matrix (homography matrix) with 8 coefficients
        pre_H = torch.tensor([
            [-0.204835137, -3.09995252, 79.9098762, 
             -2.94687413, 70.6836681, -0.0467392998, 
             0.0, 0.0]
        ], dtype=torch.float32).to(gt_pts.device).repeat(gt_pts.size(0), 1)

        # Add a 1.0 to the transformation coefficients to make it a 3x3 matrix
        # Assuming output has 8 coefficients
        # If only 6 are used, adjust accordingly
        transformation_coefficients = torch.cat([
            transformation_coefficients, 
            torch.ones(transformation_coefficients.size(0), 0).to(gt_pts.device)  # No additional coefficients
        ], dim=-1)

        # Construct the 3x3 homography matrix
        H = self._construct_homography_matrix(transformation_coefficients)

        # Transform the ground truth points
        transformed_pts = torch.bmm(H, gt_pts.permute(0, 2, 1))  # [batch_size, 3, num_points]

        # Fit a quadratic polynomial (2nd order) to the transformed points
        X, Y = self._fit_quadratic_polynomial(transformed_pts)

        # Compute the loss using the fitted values and transformed back points
        loss = self._compute_loss(X, Y, H, gt_pts)

        return loss

    def _construct_homography_matrix(self, transformation_coefficients):
        """
        Construct a 3x3 homography matrix from the coefficients
        :param transformation_coefficients: Predicted coefficients
        :return: 3x3 homography matrix
        """
        batch_size = transformation_coefficients.size(0)
        H = torch.zeros((batch_size, 3, 3), dtype=torch.float32).to(transformation_coefficients.device)

        # Fill the homography matrix with the coefficients
        H[:, 0, 0] = transformation_coefficients[:, 0]  # a
        H[:, 0, 1] = transformation_coefficients[:, 1]  # b
        H[:, 0, 2] = transformation_coefficients[:, 2]  # c
        H[:, 1, 0] = transformation_coefficients[:, 3]  # d
        H[:, 1, 1] = transformation_coefficients[:, 4]  # e
        H[:, 1, 2] = transformation_coefficients[:, 5]  # f
        H[:, 2, 0] = transformation_coefficients[:, 6]  # g
        H[:, 2, 1] = transformation_coefficients[:, 7]  # h
        H[:, 2, 2] = 1.0  # last element is 1

        return H

    def _fit_quadratic_polynomial(self, transformed_pts):
        """
        Fit a quadratic polynomial to the transformed points
        :param transformed_pts: Points after applying the transformation
        :return: X and Y values after fitting the polynomial
        """
        Y = transformed_pts[:, 1, :] / transformed_pts[:, 2, :]
        X = transformed_pts[:, 0, :] / transformed_pts[:, 2, :]
        Y_One = torch.ones_like(Y)

        # Stack the Y values for a quadratic polynomial fitting (Y^2, Y, 1)
        Y_stack = torch.stack([Y ** 2, Y, Y_One], dim=-1)

        # Solve for the polynomial weights using least squares
        YTY = torch.bmm(Y_stack.transpose(1, 2), Y_stack)
        YTX = torch.bmm(Y_stack.transpose(1, 2), X.unsqueeze(-1))
        w = torch.bmm(torch.inverse(YTY), YTX)

        # Calculate the predicted x values using the polynomial
        x_preds = torch.bmm(Y_stack, w).squeeze(-1)

        return x_preds, Y

    def _compute_loss(self, x_preds, Y, H, gt_pts):
        """
        Compute the loss between predicted points and original ground truth
        :param x_preds: Predicted X values
        :param Y: Y values used in polynomial fitting
        :param H: Homography matrix
        :param gt_pts: Original ground truth points
        :return: Mean squared error loss
        """
        # Construct the predicted points using the fitted polynomial
        lane_trans_pred = torch.stack([x_preds * gt_pts[:, :, 2], Y * gt_pts[:, :, 2], gt_pts[:, :, 2]], dim=1)

        # Transform back to the original space using the inverse of the homography matrix
        H_inv = torch.inverse(H)
        lane_trans_back = torch.bmm(H_inv, lane_trans_pred)

        # Calculate the mean squared error loss
        loss = torch.mean((gt_pts[:, :, :2] - (lane_trans_back[:, :2, :] / lane_trans_back[:, 2:3, :])) ** 2)

        return loss



